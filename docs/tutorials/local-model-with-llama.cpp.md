# Example: Local Model Setup with Llama.cpp
*Getting and setting up a local model with llama.cpp and integrating it with the AI Assistant application.*

## Overview

This guide will walk you through setting up a local model using llama.cpp, which provides an OpenAI-compatible API server that can be easily integrated with the AI Assistant application's existing provider system.

## Prerequisites

- Docker with GPU support
- NVIDIA GPU with CUDA support
- Sufficient disk space for models (typically 4-8GB per model)
- At least 8GB RAM for model loading

## Docker Setup

Docker is used to ensure a more simple and consistent setup across machines.

Below is how to set up Docker with CUDA support on Ubuntu. Adapted from the [Docker Documentation](https://docs.docker.com/engine/install)

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

# Properly setting up permissions
sudo usermod -aG docker $USER 

# Reset the computer, or run this command for this terminal
newgrp docker
```

## Setting up a Models Directory

Determine a directory for models on your machine. This example uses `/home/ubuntu/llama-models`:

```bash
MODEL_DIR=/home/ubuntu/llama-models
mkdir -p $MODEL_DIR
```

## Model Conversion Script

Create a Python script to use Unsloth to download and convert a model to GGUF format. Save it as `$MODEL_DIR/convert_model.py`:

```python
# Model conversion script using Unsloth
# This script downloads a model and converts it to GGUF format for use with llama.cpp

from unsloth import FastVisionModel
import torch
import os

# Model configuration
model_name = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
output_directory = "converted_model"
quant_method = "q4_0"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

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
print(f"Model saved in: {output_directory}")
```

## Model Conversion with Unsloth

Unsloth makes consumer models smaller, more efficient, and able to be used with llama.cpp.

```bash
# Start and enter the Unsloth Docker container
docker run --rm -it --gpus all \
  -v $MODEL_DIR:/data \
  -v $MODEL_DIR/hf_cache:/root/.cache/huggingface \
  --entrypoint "/bin/bash" \
  -u root \
  unsloth/unsloth:latest-cu121-py310

# Inside the container:
cd /data

# May need to do this until the Unsloth Docker image is updated
# Ignore warnings.
pip install --upgrade transformers

# Run the conversion script
# Default password is: unsloth when prompted
python3 convert_model.py

# After conversion, exit the container
exit
```

## Finding the Converted Model

After the conversion completes, you need to find the actual GGUF file name:

```bash
# List the converted model files
ls -la $MODEL_DIR/converted_model/

# You should see a file with a name like:
# qwen3-vl-8b-thinking.Q4_0.gguf
```

Note the exact filename as you'll need it for the next step.

## Serving with Llama.cpp

Now we'll serve the model using llama.cpp with OpenAI-compatible API server mode:

```bash
# Get the actual model filename (replace with your actual filename)
MODEL_FILE=$(ls $MODEL_DIR/converted_model/*.gguf | head -n 1 | xargs basename)

# Start the llama.cpp server with OpenAI-compatible API
docker run -d --name llama-server \
  --gpus all \
  -p 8089:8080 \
  -v $MODEL_DIR/converted_model:/models \
  ghcr.io/ggml-org/llama.cpp:full-cuda \
  -m /models/$MODEL_FILE \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key llama-cpp-key \
  -c 4096 \
  -ngl 35

# Check if the server is running
docker logs llama-server
```

The server will now be available at `http://localhost:8089` with an OpenAI-compatible API endpoint.

## Verifying the Server

Test that the server is working correctly:

```bash
# List available models
curl http://localhost:8089/v1/models \
  -H "Authorization: Bearer llama-cpp-key"

# Test a simple completion
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer llama-cpp-key" \
  -d '{
    "model": "'$MODEL_FILE'",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'
```

## Integrating with the AI Assistant

Now we'll configure the AI Assistant application to use our local llama.cpp server.

### Option 1: Using Environment Variables

Add the following to your `.env` file:

```bash
# =============================================================================
# Local Llama.cpp Model Configuration
# =============================================================================

# Enable OpenAI-compatible provider for llama.cpp
OPENAI_COMPATIBLE_ENABLED=true
OPENAI_COMPATIBLE_API_KEY=llama-cpp-key
OPENAI_COMPATIBLE_BASE_URL=http://localhost:8089/v1
OPENAI_COMPATIBLE_DEFAULT_MODEL=<REPLACE_WITH_ACTUAL_MODEL_FILENAME>

# Set as preferred provider
PREFERRED_PROVIDER=openai_compatible

# Keep OpenRouter as fallback (optional)
OPENROUTER_API_KEY=your_openrouter_api_key_here
ENABLE_FALLBACK=true
```

Replace `<REPLACE_WITH_ACTUAL_MODEL_FILENAME>` with the actual filename from the conversion step (e.g., `qwen3-vl-8b-thinking.Q4_0.gguf`).

### Option 2: Using Docker Compose

If you're using Docker Compose, add the llama.cpp server to your `docker-compose.yml`:

```yaml
services:
  # ... existing services ...

  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:full-cuda
    container_name: ai-assistant-llama-server
    command: >
      -m /models/model.gguf
      --host 0.0.0.0
      --port 8080
      --api-key llama-cpp-key
      -c 4096
      -ngl 35
    volumes:
      - ./models:/models
    ports:
      - "8089:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - ai-assistant-network
    restart: unless-stopped
    profiles:
      - local-model

  # Update your main application to use the llama server
  ai-assistant:
    # ... existing configuration ...
    environment:
      # ... existing environment variables ...
      - OPENAI_COMPATIBLE_BASE_URL=http://llama-server:8080/v1  # Use Docker network
      - OPENAI_COMPATIBLE_API_KEY=llama-cpp-key
    depends_on:
      # ... existing dependencies ...
      - llama-server
```

Then update your `.env` file:

```bash
# Use the llama.cpp server in Docker
OPENAI_COMPATIBLE_BASE_URL=http://llama-server:8080/v1
OPENAI_COMPATIBLE_API_KEY=llama-cpp-key
OPENAI_COMPATIBLE_DEFAULT_MODEL=model.gguf
PREFERRED_PROVIDER=openai_compatible
```

## Testing the Integration

Start the AI Assistant application and verify that it's using your local model:

```bash
# If using Docker Compose with the local-model profile
docker compose --profile local-model up

# Or if running the application directly
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test the integration with a simple API call:

```bash
# Test the application with your local model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai_compatible:<REPLACE_WITH_ACTUAL_MODEL_FILENAME>",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'
```

Or test via the Chainlit interface at `http://localhost:8000/chainlit`.

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure the model filename in your configuration matches exactly
   - Check that the llama.cpp server can access the model file

2. **CUDA out of memory**
   - Reduce the number of GPU layers (`-ngl` parameter)
   - Use a smaller model or more aggressive quantization

3. **Connection refused**
   - Ensure the llama.cpp server is running
   - Check that the port (8089) is not blocked by firewall

4. **Slow responses**
   - Increase the number of GPU layers if you have sufficient VRAM
   - Ensure your GPU has enough memory for the model

### Debugging Commands

```bash
# Check server logs
docker logs llama-server

# Check server status
curl http://localhost:8089/health

# List loaded models
curl http://localhost:8089/v1/models \
  -H "Authorization: Bearer llama-cpp-key"

# Check GPU usage
nvidia-smi
```

## Performance Optimization

1. **GPU Layers**: Adjust `-ngl` parameter based on your VRAM
   - 8GB VRAM: 20-30 layers
   - 16GB VRAM: 35+ layers
   - 24GB+ VRAM: All layers

2. **Context Size**: Adjust `-c` parameter based on your needs
   - Default: 4096 tokens
   - Increase for longer conversations

3. **Quantization**: Choose the right balance
   - `q4_0`: Good balance of quality and size
   - `q5_0`: Better quality, larger size
   - `q8_0`: Best quality, largest size

## Alternative Models

You can use different models by modifying the `model_name` in the conversion script:

```python
# Popular alternatives
model_name = "microsoft/DialoGPT-medium"  # For conversations
model_name = "codellama/CodeLlama-7b-hf"  # For code
model_name = "mistralai/Mistral-7B-v0.1"  # General purpose
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Chat optimized
```

## Conclusion

You now have a local model running with llama.cpp that's fully integrated with the AI Assistant application. This setup gives you:

- Privacy and data security
- No API costs
- Fast responses (limited only by your hardware)
- Full control over model selection and configuration

The setup can be further customized with different models, quantization levels, and hardware configurations to best suit your needs.