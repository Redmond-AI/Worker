import torch
from queue import Queue
import base64
from PIL import Image
from .trt.models import CLIP, UNet, VAE
import numpy as np
import json
from polygraphy import cuda
import time
import torch
from transformers import CLIPTokenizer, CLIPTextModel
import tensorrt as trt
from .trt.utilities import Engine, TRT_LOGGER, get_model_path
import traceback
from io import BytesIO
import logging
from .lpw.enc import *
import os
import random
import diffusers
import copy

log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.logging.basicConfig(level=getattr(logging, log_level),
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.logging.getLogger()

def image_generator(
    config,
    request_queue: Queue,
    image_queue: Queue,
    command_queue: Queue,
    ):

    verbose = True
    max_batch_size = 4
    device = 'cuda'

    unet_structure = {
    'unet': UNet(fp16=True, device=device, verbose=verbose, max_batch_size=max_batch_size, hf_token=""),
    }

    unet_engine = {}
    unet_map = {}
    unet_count = 0
    for entry in config['models']:
        unet_count += 1
        unet_map[entry["alias"]] = {}
        unet_map_entry = unet_map[entry["alias"]]
        unet_map_entry['struct'] = copy.deepcopy(unet_structure)
        unet_map_entry['engine'] = copy.deepcopy(unet_engine)
        unet_map_entry['config'] = copy.deepcopy(entry)
    print(f"Detected {unet_count} unets")

    vae_engine = {}
    vae_map = {}
    vae_count = 0
    vae_structure = {
        'de_vae': VAE(device=device, verbose=verbose, max_batch_size=max_batch_size, hf_token=""),
    }

    for entry in config['vae']:
        vae_count += 1
        vae_map[entry["alias"]] = {}
        vae_map_entry = vae_map[entry["alias"]]
        vae_map_entry['struct'] = copy.deepcopy(vae_structure)
        vae_map_entry['engine'] = copy.deepcopy(vae_engine)
        vae_map_entry['config'] = copy.deepcopy(entry)
    print(f"Detected {vae_count} VAEs")

    """
    Something like

    model_map = {
        "ANYTHING-V3": {
            "struct": {
                'clip': CLIP(),
                'unet': UNet(),
                #'de_vae': VAE(),
            }
            "engine": {},
            "config": {
                "alias": "ANYTHING-V3",
                "model_path": "/workspace/static/trt/anything-v3.0"
            }
        }
    }
    
    """

    # if "extras" in config:
    #     if "en_vae" in config['extras']:
    #         if 'subfolder' in config['extras']['en_vae']:
    #             subfolder = config['extras']['en_vae']['subfolder']
    #         else:
    #             subfolder = None
    #         models['en_vae'] = AutoencoderKL.from_pretrained(config['extras']['en_vae']['path'], 
    #         subfolder=subfolder)

    stream = cuda.Stream()

    for vae_entry in vae_map:
        print(vae_entry)
        vae_path = vae_map[vae_entry]['config']['model_path']
        indiv_engine = Engine(vae_path)
        indiv_engine.activate()
        vae_map[vae_entry]['engine']['de_vae'] = indiv_engine

    for unet_entry in unet_map:
        print(unet_entry)
        unet_path = unet_map[unet_entry]['config']['model_path']
        indiv_engine = Engine(unet_path)
        indiv_engine.activate()
        unet_map[unet_entry]['engine']['unet'] = indiv_engine

    print(unet_map)
    print("#####")
    print(vae_map)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    sch_source = "tensorrt/AbyssOrangeMix2_nsfw"

    schedulers = {
        "DDIM": diffusers.DDIMScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "DEIS": diffusers.DEISMultistepScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "DPM2": diffusers.KDPM2DiscreteScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "DPM2-A": diffusers.KDPM2AncestralDiscreteScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "EULER-A": diffusers.EulerAncestralDiscreteScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "EULER": diffusers.EulerDiscreteScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "HEUN": diffusers.DPMSolverMultistepScheduler.from_pretrained(sch_source, subfolder="scheduler", solver_type="heun"),
        "DPM++": diffusers.DPMSolverMultistepScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "DPM": diffusers.DPMSolverMultistepScheduler.from_pretrained(sch_source, subfolder="scheduler", algorithm_type="dpmsolver"),
        "PNDM": diffusers.PNDMScheduler.from_pretrained(sch_source, subfolder="scheduler"),
        "SING-DPM": diffusers.DPMSolverSinglestepScheduler.from_pretrained(sch_source, subfolder="scheduler"),
    }

    def runEngine(model_name, feed_dict, engine_dict):
        indiv_engine = engine_dict[model_name]
        return indiv_engine.infer(feed_dict, stream)

    def imgq(status: str, content):
        response = {
            "status": status,
            "content": content
        }
        image_queue.put(response)

    lpw_pipe = LongPromptWeightingPipeline(
        tokenizer=tokenizer,
        text_encoder=CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    )

    logger.info("Prepartions ready")
    logger.debug(schedulers)

    command_queue.put({"status": "ready"})

    while True:
        request = request_queue.get()
        data = request
        try:
            start_time = time.time()
            logger.info("got request!", data)
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
            # img = data['img'] if 'img' in data else None
            # strength = data['strength'] if 'strength' in data else None

            #TODO: a little bit confusing
            logger.debug("scheduler section")
            logger.debug(scheduler)
            if scheduler in schedulers:
                scheduler = schedulers[scheduler]
                scheduler.set_timesteps(steps, device=device)
            else:
                imgq('fail', f'scheduler {scheduler} not found')
                continue

            logger.debug("model section")
            if model in unet_map:
                curr_model = unet_map[model]
            else:
                imgq('fail', f'model {model} not found')
                print(unet_map)
                continue

            if vae in vae_map:
                curr_vae = vae_map[vae]
            else:
                imgq('fail', f'vae {vae} not found')
                continue

            curr_engine_dict = curr_model['engine']
            curr_vae_dict = curr_vae['engine']

            batch_size = 1
            
            # Spatial dimensions of latent tensor
            latent_height = image_height // 8
            latent_width = image_width // 8

            logger.debug("alloc buffer")
            # Allocate buffers for TensorRT engine bindings

            for model_name, obj in curr_model['struct'].items():
                curr_engine_dict[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=device)

            for model_name, obj in curr_vae['struct'].items():
                curr_vae_dict[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=device)

            # for model_name, obj in models.items():
            #     if model_name == 'en_vae':
            #         continue
            #     engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=device)

            logger.debug("seed gen")
            # Seeds
            generator = None

            if seed == -1:
                seed = random.randint(1, 100000000000000) #<- 16 characters maximun
            generator = torch.Generator(device="cuda").manual_seed(seed)

            preparation_time = time.time()

            logger.debug("running pipe")
            # Run Stable Diffusion pipeline
            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER) as runtime:
                # latents need to be generated on the target device
                unet_channels = 4 # unet.in_channels

                text_embeddings = lpw_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=cfg,
                    num_images_per_prompt=1,
                    max_embeddings_multiples=1
                )
                text_embeddings = text_embeddings.to(dtype=torch.float16)

                logger.debug("novo")
                
                logger.debug(text_embeddings)
                logger.info(text_embeddings.shape)

                dtype = text_embeddings.dtype

                clip_time = time.time()
                #prob add benchmark for latent gen?
                latents_shape = (batch_size, unet_channels, latent_height, latent_width)
                logger.debug("latents_shape:", latents_shape)
                latents_dtype = torch.float32 # text_embeddings.dtype
                latents = torch.randn(latents_shape, device=device, dtype=latents_dtype, generator=generator)

                # Scale the initial noise by the standard deviation required by the scheduler
                latents = latents * scheduler.init_noise_sigma

                torch.cuda.synchronize()                

                logger.debug("denoising")
                for step_index, timestep in enumerate(scheduler.timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

                    # predict the noise residual
                    dtype = np.float16
                    if timestep.dtype != torch.float32:
                        timestep_float = timestep.float()
                    else:
                        timestep_float = timestep
                    sample_inp = cuda.DeviceView(ptr=latent_model_input.data_ptr(), shape=latent_model_input.shape, dtype=np.float32)
                    timestep_inp = cuda.DeviceView(ptr=timestep_float.data_ptr(), shape=timestep_float.shape, dtype=np.float32)
                    embeddings_inp = cuda.DeviceView(ptr=text_embeddings.data_ptr(), shape=text_embeddings.shape, dtype=dtype)
                    noise_pred = runEngine('unet', {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp}, curr_engine_dict)['latent']

                    # Perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

                    # Denoise?
                    if data['scheduler'] in ['DEIS', 'DPM2', 'HEUN', 'DPM++', 'DPM', 'PNDM', 'SING-DPM']:
                        latents = scheduler.step(
                                model_output=noise_pred, 
                                timestep=timestep, 
                                sample=latents).prev_sample
                    else:
                        latents = scheduler.step(
                                model_output=noise_pred, 
                                timestep=timestep, 
                                sample=latents,
                                generator=generator).prev_sample

                denoising_time = time.time()

                logger.debug("finished")
                latents = 1. / 0.18215 * latents
                
                sample_inp = cuda.DeviceView(ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32)
                images = runEngine('de_vae', {"latent": sample_inp}, curr_vae_dict)['images']

                vae_time = time.time()
                
                torch.cuda.synchronize()
                logger.debug("syncronized, converting to img")
                images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
                img = Image.fromarray(images[0])
                serving_time = time.time()
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                # imgq("done", {"time": serving_time - preparation_time, "seed": seed, "img": base64.b64encode(buffered.getvalue()).decode('utf-8')})
                imgq("done", {
                    "image": base64.b64encode(buffered.getvalue()).decode('utf-8'),
                    "metadata": {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "model": model,
                        "vae": vae,
                        "steps": steps,
                        "width": image_width,
                        "height": image_height,
                        "cfg": cfg,
                        "seed": seed,
                        "scheduler": data['scheduler'],
                        "compute_time": serving_time - preparation_time
                    }
                })
                benchmark_time = {
                    "PREP": preparation_time - start_time,
                    "CLIP": clip_time - preparation_time,
                    f"UNET x {steps}*": denoising_time - clip_time,
                    "VAE*": vae_time - denoising_time,
                    "SERVING": serving_time - vae_time,
                    "TOTALCOM": serving_time - preparation_time,
                    "TOTAL": serving_time - start_time,
                }
                print("Benchs: (Check notes)")
                for i in benchmark_time:
                    print('| {:^14} | {:>9.2f} ms |'.format(i, int(benchmark_time[i]*1000)))
                print(f'w{image_width} x h{image_height}')
                print('scheduler: {}'.format(data['scheduler']))

        except Exception as e:
            traceback.print_exc()
            imgq('fail', f'general exception, got {str(e)}')
            continue

