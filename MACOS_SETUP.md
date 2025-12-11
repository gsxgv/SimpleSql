# macOS Setup Guide

## Quick Assessment Summary

**Your Hardware**: MacBook Pro M2, 8GB RAM
**Status**: ⚠️ **Limited Compatibility** - Some features will not work

## Critical Issues

1. ❌ **bitsandbytes** - Does NOT work on Apple Silicon (CUDA-only)
2. ❌ **QLoRA with 4-bit** - Requires bitsandbytes, will NOT work
3. ⚠️ **8GB RAM** - Very limited for 7B model finetuning
4. ✅ **Ollama** - Fully compatible, recommended for inference

## Recommended Approach

### Option 1: Ollama-Only Workflow (RECOMMENDED for 8GB RAM)

**Best for**: Evaluation and inference

1. Use Ollama for all model operations
2. Skip local finetuning (use cloud or more powerful machine)
3. Focus on evaluation pipeline

**Steps**:
```bash
# Install Ollama
brew install ollama
ollama serve

# Pull a model (smaller models recommended for 8GB)
ollama pull sqlcoder-7b
# OR use a smaller model:
ollama pull mistral:7b-instruct-q4_0  # Quantized, smaller memory footprint

# Run evaluation
python scripts/evaluate_baseline.py --config config-macos.yaml
```

### Option 2: Limited Local Finetuning

**Best for**: Learning/experimentation with small models

**Requirements**:
- Use `config-macos.yaml` (optimized settings)
- Use LoRA (not QLoRA)
- Very small batch size (1)
- Consider smaller models (1B-3B parameters)

**Steps**:
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install macOS-compatible requirements
pip install -r requirements-macos.txt

# Use macOS config
python scripts/finetune_model.py --config config-macos.yaml
```

## Installation Steps

### 1. Install PyTorch with MPS Support

```bash
pip install torch torchvision torchaudio
```

Verify MPS is available:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

### 2. Install macOS-Compatible Requirements

```bash
pip install -r requirements-macos.txt
```

**Note**: `requirements-macos.txt` excludes `bitsandbytes` which doesn't work on Apple Silicon.

### 3. Use macOS-Optimized Configuration

Copy the macOS config:
```bash
cp config-macos.yaml config.yaml
```

Or use it explicitly:
```bash
python scripts/evaluate_baseline.py --config config-macos.yaml
```

### 4. Install Ollama

```bash
brew install ollama
ollama serve
```

## Configuration Differences

### macOS Config (`config-macos.yaml`)

- `method: "lora"` (not "qlora")
- `load_in_4bit: false` (cannot use)
- `load_in_8bit: false` (cannot use)
- `batch_size: 1` (reduced from 4)
- `lora_r: 8` (reduced from 16)
- `max_seq_length: 1024` (reduced from 2048)
- `gradient_accumulation_steps: 8` (increased to maintain effective batch size)

## Memory Management Tips

With 8GB RAM, you need to be careful:

1. **Close other applications** during finetuning
2. **Use smaller models** (1B-3B instead of 7B)
3. **Use Ollama** for inference (handles memory efficiently)
4. **Monitor memory usage**:
   ```bash
   # In another terminal
   top -l 1 | grep "PhysMem"
   ```

## Expected Performance

### Inference (via Ollama)
- ✅ **Fast** - Ollama handles memory efficiently
- ✅ **Stable** - No memory issues
- ✅ **Recommended** - Best option for 8GB RAM

### Finetuning (Local)
- ⚠️ **Slow** - Limited by CPU/MPS and memory
- ⚠️ **May fail** - 7B models may cause OOM errors
- ⚠️ **Limited** - Very small batch sizes required

## Troubleshooting

### "bitsandbytes not found" Error
**Solution**: This is expected on Apple Silicon. Use `config-macos.yaml` which disables quantization.

### Out of Memory Errors
**Solutions**:
1. Reduce `batch_size` to 1
2. Reduce `max_seq_length` to 512 or 1024
3. Use smaller models
4. Use Ollama instead of local finetuning

### MPS Backend Not Available
**Solution**: 
```bash
pip install --upgrade torch torchvision torchaudio
```

### Model Too Large
**Solutions**:
1. Use quantized models via Ollama
2. Use smaller base models (1B-3B parameters)
3. Consider cloud finetuning

## Alternative: Cloud Finetuning

If local finetuning is too limited:

1. **Google Colab** (free GPU)
2. **Kaggle** (free GPU)
3. **AWS/GCP** (paid)

Finetune in cloud, then:
- Download finetuned model
- Convert to GGUF
- Import to Ollama
- Evaluate locally

## Summary

**For your 8GB M2 MacBook Pro:**

✅ **DO**:
- Use Ollama for inference
- Run evaluation scripts
- Use `config-macos.yaml`
- Consider smaller models

❌ **DON'T**:
- Try to use bitsandbytes
- Use QLoRA/4-bit quantization
- Finetune 7B+ models locally
- Expect fast finetuning

**Best Workflow**: Use Ollama for everything, finetune in cloud if needed.

