# Use the base image
FROM ubuntu:20.04

ENV NVIDIA_VISIBLE_DEVICES all
# Install required packages
RUN apt-get update && apt-get install -y sudo wget\
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN sudo apt-get update
RUN sudo apt-get install -y git

RUN mkdir /node
# Set working directory
WORKDIR /node
RUN mkdir -p /home/storage

# Copy application files into the container
COPY ./ /node

# Run the set of commands
RUN cd /node

# ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
ENV GIT_PYTHON_REFRESH=quiet

RUN /bin/bash -c "chmod +x get-python.sh setup.sh link_folders.sh"

RUN /bin/bash -c "./get-python.sh"

RUN /bin/bash -c "./setup.sh"

RUN /bin/bash -c "./link_folders.sh"

RUN /bin/bash -c "python3 -m pip install -r ./stable-diffusion-webui/requirements_versions.txt"
RUN /bin/bash -c "python3 -m pip install -r ./ControlNet/requirements.txt"
RUN /bin/bash -c "python3 -m pip install clip gdown triton"

# Expose a port if needed
EXPOSE 5000

# ARG CUDA_DEVICE_ID
ENV DEVICE_ID = 0
CMD sh -c 'python3 ./main_webui.py --device-id=$DEVICE_ID'