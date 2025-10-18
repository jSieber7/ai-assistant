# Example: Local Model Setup
*Getting and setting up a local model with the application.*

## Docker
Docker is used to ensure a more simple and consistent setup across machines.

Below is how to setup up a docker in ubuntu.
Adapted from the [Docker Documentation](https://docs.docker.com/engine/install)

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install the latest version of Docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Ensuring Docker has CUDA capabilities
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Properly setting up permissons
sudo usermod -aG docker $USER 

# Reset the computer, or run this command for this terminal
newgrp docker
```

## Setting up a Models directory
Determine a directory for models on your machine. I chose /home/ubuntu/llama-models for my setup.

```bash
MODEL_DIR=/home/ubuntu/llama-models
cd $MODEL_DIR
mkdir $MODEL_DIR/hf_models
```

## Conversion Python Script
A script to use Unsloth to gather and convert a model to one we need. Save it in the model directory you chose.
You can change the model and quantization method at your discretion.

```python
# Written by Gemini-Pro-2.5

from unsloth import FastVisionModel
import torch

model_name = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"

# --- CHANGED: This is now a directory name, not a filename ---
output_directory = "model_for_gguf"

quant_method = "q4_0"

print(f"Loading model: {model_name}")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True,
    dtype = None,
    device_map = "auto",
)

print(f"\nSaving model to GGUF format in directory '{output_directory}'")
# The function will create this directory and put the GGUF inside.
model.save_pretrained_gguf(output_directory, tokenizer, quantization_method = quant_method)

print("\nConversion complete!")
```

## Unsloth model conversion
Unsloth makes consumer models smaller, more efficent, and able to be used with the local LLM provider Llama.ccp 

```bash
# Start and enter the docker
docker run --rm -it --gpus all \
  -v /home/ubuntu/llama-models:/data \
  -v $MODEL_DIR/hf_models:/root/.cache/huggingface \
  --entrypoint "/bin/bash" \
  -u root \
  unsloth/unsloth:latest-cu121-py310

# May need to do this until the Unsloth Docker image is updated
# Ignore warnings.
pip install --upgrade transformers

# Run the conversion script
# Default password is: unsloth when prompted
python3 /data/convert_with_unsloth.py
```

## Serving with Llama.ccp
Serves the model we have downloaded and converted via Unsloth
This makes the model able to be utilized by our app
We use port to avoid conflicts

```bash
docker run -d --name llama-server \
  --gpus all \
  -p 8089:8080 \
  -v $MODEL_DIR/models \
  ghcr.io/ggml-org/llama.cpp:full-cuda \
  -s -m /models/qwen3-vl-8b-q4.gguf --host 0.0.0.0 --port 8080 -ngl 35
```