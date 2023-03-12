import logging
import os
import sys
from os.path import join, abspath, dirname
sys.path.append(join(dirname(abspath(__file__)), "stable-diffusion-webui"))
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules import shared
import modules.script_callbacks
from modules.shared import sd_upscalers, opts, parser
from modules.sd_models import checkpoint_alisases
from modules.sd_vae import vae_dict
from modules.call_queue import queue_lock
from webui import initialize
import traceback
import time
from io import BytesIO
import base64

log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level),
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

initialize()
modules.script_callbacks.before_ui_callback()

txt2img_processing = StableDiffusionTxt2ImgProcessingAPI()

def paramatersToMetadata(data, serving_time, preparation_time):
    prompt = data['prompt']
    negative_prompt = data['negative_prompt']
    image_height = data['height']
    image_width = data['width']
    scheduler = data['scheduler']
    steps = data['steps']
    cfg = data['cfg']
    seed = data['seed']
    model = data['model']
    vae = data['vae']

    return {"prompt": prompt, "negative_prompt": negative_prompt, "model": model, "vae": vae,
                        "steps": steps,
                        "width": image_width,
                        "height": image_height,
                        "cfg": cfg,
                        "seed": seed,
                        "scheduler": scheduler, "compute_time": serving_time - preparation_time}

def parametersToProcessing(data):
    prompt = data['prompt']
    negative_prompt = data['negative_prompt']
    image_height = data['height']
    image_width = data['width']
    scheduler = data['scheduler']
    steps = data['steps']
    cfg = data['cfg']
    seed = data['seed']
    model = data['model']
    vae = data['vae']
    print(checkpoint_alisases)
    if model not in checkpoint_alisases:
        raise Exception("model not supported")
    print(vae_dict)
    if vae not in vae_dict:
        raise Exception("vae not supported")

    return txt2img_processing.copy(update={
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": image_height,
        "width": image_width,
        "sampler_name": scheduler,
        "steps": steps,
        "cfg_scale": cfg,
        "seed": seed,
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae
        }
    })



def image_generator(config, request_queue, image_queue, command_queue):
    verbose = True
    max_batch_size = 4
    device = 'cuda'

    def imgq(status: str, content):
        response = {
            "status": status,
            "content": content
        }
        image_queue.put(response)

    command_queue.put({"status": "ready"})
    while True:
        request = request_queue.get()
        data = request
        try:
            start_time = time.time()
            logger.info("got request!", data)
            processing_request = parametersToProcessing(data)
            args = vars(processing_request)
            args.pop('script_name', None)

            send_images = args.pop('send_images', True)
            args.pop('save_images', None)
            preparation_time = time.time()
            with queue_lock:
                p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
                p.outpath_grids = opts.outdir_txt2img_grids
                p.outpath_samples = opts.outdir_txt2img_samples

                shared.state.begin()
                processed = process_images(p)
                shared.state.end()
            # b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
            serving_time = time.time()
            buffered = BytesIO()
            processed.images[0].save(buffered, format="PNG")
            imgq("done", {"image": base64.b64encode(buffered.getvalue()).decode('utf-8'), "metadata": paramatersToMetadata(data, serving_time, preparation_time)})
        except Exception as e:
            traceback.print_exc()
            imgq('fail', f'general exception, got {str(e)}')
            continue

