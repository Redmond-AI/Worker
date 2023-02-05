from diffusers import StableDiffusionPipeline
from safetensors import torch as tc
import torch

pipe = StableDiffusionPipeline.from_pretrained("Linaqruf/anything-v3.0", torch_dtype=torch.float16)

prompt = "masterpiece, high quality, 1girl, black military uniform, black military hat, small breasts, [swastika armband], simple background"
pipe.to("cuda")
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon_wo.png")

hf_lora = torch.load('/workspace/india/node/threads/experimental/lora/pytorch_lora_weights.bin', map_location="cpu")
tmp_model = tc.load_file("/workspace/test.safetensors")

print(hf_lora)
print("#####")
print("END OF HF LORA AND START OF TMP MODEL")
print("####")
print(tmp_model)

pipe.unet.load_attn_procs(tmp_model)

prompt = "masterpiece, high quality, 1girl, black military uniform, black military hat, small breasts, [swastika armband], simple background"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon_wi.png")