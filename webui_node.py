
import os
import sys
from os.path import join, abspath, dirname
sys.path.append(join(dirname(abspath(__file__)), "stable-diffusion-webui"))

from modules import paths, shared, modelloader
from webui import initialize
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
model_url = None

initialize()


model_list = modelloader.load_models(
    model_path=model_path, 
    model_url=model_url, 
    command_path=shared.cmd_opts.ckpt_dir, 
    ext_filter=[".ckpt", ".safetensors"], 
    download_name="v1-5-pruned-emaonly.safetensors", 
    ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

print(model_list)
