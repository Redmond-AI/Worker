cd ~/
sudo apt update
sudo apt install gcc
sudo apt install make
wget https://us.download.nvidia.com/tesla/525.85.12/NVIDIA-Linux-x86_64-525.85.12.run
sudo chmod +x NVIDIA-Linux-x86_64-525.85.12.run
sudo ./NVIDIA-Linux-x86_64-525.85.12.run
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
##git clone https://Redmond-AI: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@github.com/pixelizeai/node.git
# git clone https://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@github.com/Redmond-AI/Worker.git --single-branch --branch stable-diffusion-integration
# cd Worker
# git checkout stable-diffusion-integration
# git submodule update --init --recursive
# cd stable-diffusion-webui
python3 -m pip install --user --upgrade aws-sam-cli 
sudo apt install libgl1-mesa-glx
# python3 launch.py
# kill webui instance after it launches
# cd ~/
python3 -m pip install flask-socketio
python3 -m pip install gunicorn

# mkdir stable-diffusion-webui
# mkdir cfg
# cd stable-diffusion-webui

# mkdir Codeformer
# mkdir ESRGAN
# mkdir hypernetworks
# mkdir Lora
# mkdir Stable-diffusion
# mkdir VAE

# mkdir deepbooru
# mkdir GFPGAN
# mkdir LDSR
# mkdir RealESRGAN
# mkdir SwinIR
# mkdir VAE-approx


# cd ~/
# mkdir storage
# mkdir storage/vae
# mkdir storage/bundles
# mkdir storage/embeds
# mkdir storage/loras
# cd storage/vae
# wget https://models2.us-east-1.linodeobjects.com/pts/Anything-V3.0.vae.pt
# wget https://www.dropbox.com/s/y7i22ovk5pqlyzz/kl-f8-anime2.ckpt
# wget https://www.dropbox.com/s/q664tvfqarfs2pc/vae-ft-mse-840000-ema-pruned.ckpt
# wget https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt
# mv orangemix.vae.pt nai.vae.pt
# cd ../bundles
# wget https://models2.us-east-1.linodeobjects.com/ckpt/anythingV3_fp16.ckpt
# wget https://www.dropbox.com/s/o1o0nbq3kjs6r5y/URPM-BE-10B.safetensors
# wget https://civitai.com/api/download/models/17233 —
# mv 17233 AbyssOrangeMix3.safetensors
# wget https://civitai.com/api/download/models/8145  —-
# mv 8145 AresMix.safetensors
# wget https://huggingface.co/naonovn/chilloutmix_NiPrunedFp32Fix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors
# wget https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix3/AOM3A1B_orangemixs.safetensors
# wget https://www.dropbox.com/s/x38rzgulnzq37lw/homoerotic_v2.safetensors
# wget https://models2.us-east-1.linodeobjects.com/safetensors/realisticVisionV20_v20.safetensors
# cd ../embeds
# wget https://civitai.com/api/download/models/8954
# wget https://civitai.com/api/download/models/3113
# wget https://huggingface.co/yesyeahvh/bad-hands-5/resolve/main/bad-hands-5.pt
# wget https://huggingface.co/Xynon/models/resolve/main/experimentals/TI/bad-image-v2-39000.pt
# wget https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/bad_prompt_version2.pt 
# wget https://civitai.com/api/download/models/9208
# wget https://huggingface.co/yesyeahvh/mgm/resolve/main/NG_DeepNegative_V1_75T.pt
# mv 3113 bukkakAI.pt
# mv 9208 easynegative.safetensors
# mv 8954 awaitingtongue.pt
# cd ../loras
# wget https://www.dropbox.com/s/hr721x5kt7l2qwp/LORARealSquirting_squirting10.safetensors
# wget 'http://models2.us-east-1.linodeobjects.com/safetensors/Realistic Titfuck.safetensors'
# wget "https://models2.us-east-1.linodeobjects.com/safetensors/POV Doggystyle LoRA.safetensors"
# wget "https://models2.us-east-1.linodeobjects.com/safetensors/biggergirls_128.safetensors"
# cd ~/
