# Use the base image
FROM ubuntu:20.04
df
# Install required packages
RUN apt-get update && apt-get install -y sudo wget\
    # apt-get install -y wget\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /node
# Set working directory
WORKDIR /node

# Copy application files into the container
COPY ./ /node

# Run the set of commands
RUN cd /node

RUN /bin/bash -c "chmod +x get-python.sh setup.sh"

RUN /bin/bash -c "./get-python.sh"

RUN /bin/bash -c "./setup.sh"
RUN /bin/bash -c "python3 -m pip install -r ./stable-diffusion-webui/requirements_versions.txt"
# Expose a port if needed
EXPOSE 5000

ARG CUDA_DEVICE_ID=0

CMD python3 ./main_webui.py --device-id=${CUDA_DEVICE_ID}
