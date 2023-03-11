
import os
import sys
from os.path import join, abspath, dirname
sys.path.append(join(dirname(abspath(__file__)), "stable-diffusion-webui"))
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI 
from modules import paths, shared, modelloader
from webui import initialize
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
model_url = None

initialize()

txt2img_processing = StableDiffusionTxt2ImgProcessingAPI()

model_list = modelloader.load_models(
    model_path=model_path, 
    model_url=model_url, 
    command_path=shared.cmd_opts.ckpt_dir, 
    ext_filter=[".ckpt", ".safetensors"], 
    download_name="v1-5-pruned-emaonly.safetensors", 
    ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

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

    return txt2img_processing.copy(update={
        "seed": seed,
        "override_settings": {
            
        }
    })



def image_generator(config, request_queue, image_queue, command_queue):
    verbose = True
    max_batch_size = 4
    device = 'cuda'

    command_queue.put({"status": "ready"})
    while True:
        request = request_queue.get()
        data = request
        try:
            start_time = time.time()
            logger.info("got request!", data)
             
print(model_list)
