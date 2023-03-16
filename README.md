instructions

```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
git clone https://github.com/pixelizeai/node.git
# Redmond-AI / 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
Cd node
git checkout stable-diffusion-integration
git submodule update --init --recursive
cd stable-diffusion-webui
python3 -m pip install --user --upgrade aws-sam-cli 
sudo apt install libgl1-mesa-glx
python3 launch.py
# kill webui instance after it launches (ie. CTRL + C)
cd ..
python3 -m pip install flask-socketio
python3 -m pip install gunicorn

# inside of link_folders.sh, change STORAGE_FOLDER to the folder where you want to put your models
# change WEBUI_FOLDER to the folder where the webui is installed, usually node/stable-diffusion-webui
chmod +x link_folders.sh
. link_folders.sh

# models go in $STORAGE_FOLDER/bundles/
# vae go in $STORAGE_FOLDER/vae/
# loras go in $STORAGE_FOLDER/loras/ 
# embeddings go in $STORAGE_FOLDER/embeds/

#embeddings:
cd $STORAGE_FOLDER/embeds/
wget https://civitai.com/api/download/models/8954
wget https://civitai.com/api/download/models/3113
wget https://huggingface.co/yesyeahvh/bad-hands-5/resolve/main/bad-hands-5.pt
wget https://huggingface.co/Xynon/models/resolve/main/experimentals/TI/bad-image-v2-39000.pt
wget https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/bad_prompt_version2.pt 
wget https://civitai.com/api/download/models/9208
wget https://huggingface.co/yesyeahvh/mgm/resolve/main/NG_DeepNegative_V1_75T.pt
mv 3113 bukkakAI.pt
mv 9208 easynegative.safetensors
mv 8954 awaitingtongue.pt
#Models:
cd $STORAGE_FOLDER/bundles/
wget https://civitai.com/api/download/models/17233mv 17233 AbyssOrangeMix3.safetensorswget https://civitai.com/api/download/models/8145 this is aresmixmv 8145 AresMix.safetensors
#VAE:
cd $STORAGE_FOLDER/vae/
wget https://www.dropbox.com/s/y7i22ovk5pqlyzz/kl-f8-anime2.ckpt
wget https://www.dropbox.com/s/q664tvfqarfs2pc/vae-ft-mse-840000-ema-pruned.ckpt
wget https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt
mv orangemix.vae.pt nai.vae.pt

cd ~/node 
tmux 
# make sure that the config file at cfg/basic.json is correct, where:
# storage: folder where your models are stored, $STORAGE_FOLDER
# server: server of the manager
cat cfg/basic.json
python3 main_webui.py --device-id=0
#CTRL+B D to exit

```