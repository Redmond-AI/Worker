import logging
import os
import sys
from os.path import join, abspath, dirname
sys.path.append(join(dirname(abspath(__file__)), "stable-diffusion-webui"))
from modules.devices import device, get_optimal_device_name
# device = 1
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules import shared
shared.opts.sd_checkpoint_cache = 10
shared.opts.sd_vae_checkpoint_cache = 10
# shared.device = "cuda:1"
# device = 1
# shared.device = 1
# shared.cmd_opts.device_id = 1
import modules.script_callbacks
from modules.shared import sd_upscalers, opts, parser
from modules.sd_models import checkpoint_alisases
from modules.sd_vae import vae_dict
from modules.call_queue import queue_lock
from modules.upscaler import Upscaler
from modules.realesrgan_model import UpscalerRealESRGAN, get_realesrgan_models
from modules.esrgan_model import UpscalerESRGAN, get_esrgan_models
# from modules.devices import device, get_optimal_device_name
from webui import initialize
import traceback
import time
from io import BytesIO
import base64
import io
from PIL import Image,PngImagePlugin
from modules import postprocessing
import piexif
import piexif.helper
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from modules import codeformer_model
import numpy as np
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level),
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
# shared.cmd_opts.device_id = 1
# shared.device = 1
# device = 1
print(shared.weight_load_location, shared.cmd_opts.device_id, get_optimal_device_name())
initialize()
modules.script_callbacks.before_ui_callback()
print(shared.weight_load_location)
txt2img_processing = StableDiffusionTxt2ImgProcessingAPI()
img2img_processing = StableDiffusionImg2ImgProcessingAPI()

def base64_to_image(base64_string):
    # Remove data URI scheme if it exists
    if "data:" in base64_string:
        base64_string = base64_string.split(",", 1)[1]
        
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Load bytes into a PIL Image object
    image = Image.open(io.BytesIO(image_bytes))
    
    return image
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

def paramatersToMetadataUpscaler(data, serving_time, preparation_time):
    init_img = data['init_img']
    upscaler_model = data['upscaler_model']

    return {"init_img": init_img, "upscaler_model": upscaler_model, "compute_time": serving_time - preparation_time}

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
    loras = data.get('loras', [])
    embeddings = data.get('embeddings', [])

    prompt += " ".join([f"<lora:{lora['name']}:{lora['strength']}>" for lora in loras])
    for embd in embeddings:
        if embd["type"] == "positive":
            prompt += f" ,{embd['name']}:{embd['strength']}"
        else:
            negative_prompt += f" ,{embd['name']}:{embd['strength']}"

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
        },
        "override_settings_restore_afterwards": False
    })

def parametersToProcessingImg2Img(data):
    inpainting_mask_invert = data.get('inpainting_mask_invert', None)
    inpaint_full_res= data.get('inpaint_full_res', False) # bool
    inpaint_full_res_padding= data.get('inpaint_full_res_padding', False) # bool
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
    loras = data.get('loras', [])
    init_img = data.get('init_img', None)
    init_mask_inpaint = data.get('init_mask_inpaint', None)
    denoising_strength = data.get('denoising_strength')
    embeddings = data.get('embeddings', [])
    prompt += " ".join([f"<lora:{lora['name']}:{lora['strength']}>" for lora in loras])
    for embd in embeddings:
        if embd["type"] == "positive":
            prompt += f" ,{embd['name']}:{embd['strength']}"
        else:
            negative_prompt += f" ,{embd['name']}:{embd['strength']}"

    print(checkpoint_alisases)
    if model not in checkpoint_alisases:
        raise Exception("model not supported")
    print(vae_dict)
    if vae not in vae_dict:
        raise Exception("vae not supported")
    update = {
        "prompt": prompt,
        "denoising_strength": denoising_strength,
        "negative_prompt": negative_prompt,
        "height": image_height,
        "width": image_width,
        "sampler_name": scheduler,
        "steps": steps,
        "cfg_scale": cfg,
        "seed": seed,
        "init_images": [base64_to_image(init_img)],
        "inpaint_full_res":inpaint_full_res,
        "inpaint_full_res_padding": inpaint_full_res_padding,
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae
        },
        "override_settings_restore_afterwards": False
    }
    if init_mask_inpaint != None:
        update['mask'] = base64_to_image(init_mask_inpaint)
    if inpainting_mask_invert != None:
        update['inpainting_mask_invert'] = inpainting_mask_invert
    return img2img_processing.copy(update=update)


def image_generator(config, request_queue, image_queue, command_queue):
    verbose = True
    max_batch_size = 4
    # device = 'cuda'

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
            generation_type = data['generation_type'] # img2img or txt2img
            result = None
            
            if generation_type == 'upscaler':
               result= Up(data)
            else:
               result = ImageCreation(data)

            imgq("done", result)
        except Exception as e:
            traceback.print_exc()
            imgq('fail', f'general exception, got {str(e)}')
            continue
        
def ImageCreation(data):
        generation_type = data['generation_type'] # img2img or txt2img
        processing_request= None
        if generation_type == 'txt2img':
            processing_request = parametersToProcessing(data)
        elif generation_type == 'img2img':
            processing_request = parametersToProcessingImg2Img(data)
        args = vars(processing_request)
        args.pop('script_name', None)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)
        preparation_time = time.time()
        with queue_lock:
            p = None
            if generation_type == 'txt2img':
                p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            elif generation_type == 'img2img':
                p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
                
            p.outpath_grids = opts.outdir_txt2img_grids
            p.outpath_samples = opts.outdir_txt2img_samples
            shared.state.begin()
            processed = process_images(p)
            shared.state.end()
        # b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
        serving_time = time.time()
        buffered = BytesIO()
        img = processed.images[0]
        
        if generation_type == 'img2img':
            codeformer_weight = data['codeformer_weight']
            codeformer_visibility = data['codeformer_visibility']
            if codeformer_visibility != 0:
                restored_img = codeformer_model.codeformer.restore(np.array(img, dtype=np.uint8), w=codeformer_weight)
                restored_img = Image.fromarray(restored_img)
                if codeformer_visibility < 1.0:
                    img = Image.blend(img, restored_img, codeformer_visibility)
                else:
                    img = restored_img
        
        img.save(buffered, format="PNG")
        metadata = paramatersToMetadata(data, serving_time, preparation_time)
        return {'image': base64.b64encode(buffered.getvalue()).decode('utf-8'), 'metadata':metadata}


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict

def encode_pil_to_base64(image):
    print(image.tobytes())
    # stream = io.BytesIO()
    # # Save the image to the stream in JPEG format
    # image.save(stream, format='PNG')

    # # Encode the image data in base64
    # base64_data = base64.b64encode(stream.getvalue()).decode('utf-8')
    # # Print the data URI
    # return base64_data


def Up (req):
    model_name = req['upscaler_model']
    print("model NAME", model_name)
    # model_name = "R-ESRGAN General 4xV3"
    image = base64_to_image(req['init_img'])
    print(image.mode)
    # Split the image into channels
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        image = Image.merge('RGB', (r, g, b))
        print("The image is RGBA")
        print(image.mode)
    elif image.mode == "RGB":
        print("The image is RGB")
    # Create a new image without the alpha channel
    # upscaler = Upscaler(name="R-ESRGAN General 4xV3",
    # path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    # scale=4,
    # model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'))
    #['R-ESRGAN General 4xV3', 'R-ESRGAN General WDN 4xV3', 'R-ESRGAN AnimeVideo', 'R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B', 'R-ESRGAN 2x+']
    model_path = None
    UpscalerClass = None
    img = None
    models = get_realesrgan_models(None)
    realesrgan_selected_model_path = next((x.data_path for x in models if x.name == model_name), None)
    # print(realesrgan_selected_model_path)
    
    if realesrgan_selected_model_path != None:
        model_path = realesrgan_selected_model_path
        UpscalerClass = UpscalerRealESRGAN(model_path)
        img = UpscalerClass.do_upscale(image, model_path)
    else:
        model_path = get_esrgan_models(model_name)
        print(model_path)
        UpscalerClass = UpscalerESRGAN(model_path)
        img = UpscalerClass.do_upscale(image, model_path, model_name)
    
    try:
        codeformer_weight = req['codeformer_weight']
        codeformer_visibility = req['codeformer_visibility']
        if codeformer_visibility != 0:
            restored_img = codeformer_model.codeformer.restore(np.array(img, dtype=np.uint8), w=codeformer_weight)
            restored_img = Image.fromarray(restored_img)
            if codeformer_visibility < 1.0:
                img = Image.blend(img, restored_img, codeformer_visibility)
            else:
                img = restored_img
    except Exception as e:
        print("no codeformer", e)
            
    preparation_time = time.time()

    # path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    # path = "https://github.com/HyeongJu916/Boaz-SR-ESRGAN-PyTorch/blob/master/ESRGAN_4x.pth"
    # path = "https://models2.us-east-1.linodeobjects.com/upscale/4x-UltraSharp.pth"
    # ESRGAN = UpscalerESRGAN(path)
    # img = ESRGAN.do_upscale(image, path)
    # ESRGAN = UpscalerRealESRGAN(selected_model_path)
    # img = ESRGAN.do_upscale(image, selected_model_path)
    serving_time = time.time()
    metadata = paramatersToMetadataUpscaler(req, serving_time, preparation_time)
      # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format="png")
    # Turn the BytesIO object back into a bytes object
    return {'image': imgByteArr.getvalue(), 'metadata': metadata}