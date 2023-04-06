export STORAGE_FOLDER=/home/redmond/storage/
export WEBUI_FOLDER=/home/redmond/Worker/stable-diffusion-webui/

rm -dr $WEBUI_FOLDER/models/Stable-diffusion
rm -dr $WEBUI_FOLDER/models/Lora
rm -dr $WEBUI_FOLDER/models/VAE
rm -dr $WEBUI_FOLDER/embeddings

#full models (unets mostly)
ln -s $STORAGE_FOLDER/bundles/ $WEBUI_FOLDER/models/Stable-diffusion

#LORAs
ln -s $STORAGE_FOLDER/loras/ $WEBUI_FOLDER/models/Lora

#VAE
ln -s $STORAGE_FOLDER/vae/ $WEBUI_FOLDER/models/VAE

#Embeddings
ln -s $STORAGE_FOLDER/embeds/ $WEBUI_FOLDER/embeddings
