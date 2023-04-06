#!/bin/bash

# Update package list
sudo apt update

# Install Python3 and pip3
sudo apt install -y python3 python3-pip

# Set default Python3 version
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install additional packages (if needed)
# Example: sudo apt install -y <package_name>

# Verify Python3 installation
python --version
pip3 --version
