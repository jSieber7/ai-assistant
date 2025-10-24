# Ollama Reranker

This document explains how to use the Ollama-based reranker as an alternative to the custom sentence-transformers reranker.

## Overview

The Ollama Reranker provides document reranking functionality using Ollama's embedding models through LangChain. This approach eliminates the need for CUDA dependencies and allows you to use local embedding models hosted by Ollama.

## Benefits

- **No CUDA Dependency**: Eliminates the need for `nvidia-cuda-cupti-cu12` and related packages
- **Local Processing**: Uses your local Ollama instance for embedding generation
- **Flexible Model Support**: Can use any embedding model available in Ollama
- **GPU Acceleration**: If your Ollama instance is configured with GPU support, you still get GPU acceleration

## Setup

### 1. Install Ollama

First, ensure you have Ollama installed and running:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Download an Embedding Model

Download a suitable embedding model:

```bash
# Download nomic-embed-text (recommended)
ollama pull nomic-embed-text

# List available models
ollama list
```

### 3. Configure the Application

Enable the Ollama reranker in your environment:

```bash
# Disable custom reranker (uses sentence-transformers)
CUSTOM_RERANKER_ENABLED=false

# Enable Ollama reranker
OLLAMA_RERANKER_ENABLED=true

# Specify the embedding model to use
OLLAMA_RERANKER_MODEL=nomic-embed-text
```

## Configuration Options

| Setting | Default | Description |
|----------|----------|-------------|
| `OLLAMA_RERANKER_ENABLED` | `false` | Enable/disable Ollama reranker |
| `OLLAMA_RERANKER_MODEL` | `nomic-embed-text` | Embedding model to use |

## Recommended Embedding Models

- `nomic-embed-text`: Small, fast, and effective for general use
- `all-minilm`: Good balance of size and performance
- `mxbai-embed-large`: Higher quality for specialized use cases

## Usage

Once configured, the Ollama reranker works automatically in the RAG pipeline:

1. Documents are retrieved from the vector database
2. Query and documents are sent to your local Ollama instance
3. Embeddings are generated using the specified model
4. Cosine similarity is calculated to rank documents by relevance
5. Top N documents are returned with relevance scores

## Switching from Custom Reranker

To switch from the custom sentence-transformers reranker to Ollama:

1. Set `CUSTOM_RERANKER_ENABLED=false`
2. Set `OLLAMA_RERANKER_ENABLED=true`
3. Restart the application

## Performance Considerations

- **First Request**: Initial model loading may take a few seconds
- **Batch Processing**: Ollama handles multiple documents efficiently
- **Memory Usage**: Depends on the embedding model size
- **Network**: All requests go to your local Ollama instance

## Troubleshooting

### Ollama Connection Issues

If you see connection errors:

1. Verify Ollama is running: `ollama list`
2. Check the Ollama URL (default: http://localhost:11434)
3. Ensure the embedding model is downloaded: `ollama pull nomic-embed-text`

### Model Not Found

If you get "model not found" errors:

1. Check available models: `ollama list`
2. Verify the model name in `OLLAMA_RERANKER_MODEL`
3. Download the model: `ollama pull <model-name>`

### Performance Issues

For better performance:

1. Use a smaller embedding model for faster processing
2. Ensure Ollama has sufficient memory allocated
3. Consider GPU acceleration if available in your Ollama setup

## Comparison with Other Rerankers

| Feature | Custom Reranker | Ollama Reranker | Jina Reranker |
|----------|------------------|----------------|---------------|
| Local Processing | ✅ | ✅ | ❌ |
| No CUDA Dependency | ❌ | ✅ | ✅ |
| External API Required | ❌ | ❌ | ✅ |
| GPU Acceleration | ✅ | ✅ | ❌ |
| Privacy | ✅ | ✅ | ❌ |
| Setup Complexity | Medium | Low | High |