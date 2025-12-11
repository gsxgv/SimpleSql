# Small Text-to-SQL Models for macOS Finetuning

This guide recommends smaller models that can be finetuned on macOS (M2, 8GB RAM) without requiring bitsandbytes/QLoRA.

## Model Size vs Memory Requirements

| Model Size | Parameters | FP16 Memory | FP32 Memory | 8GB RAM Compatible? |
|------------|------------|-------------|-------------|---------------------|
| 1B | ~1 billion | ~2 GB | ~4 GB | ✅ Yes (comfortable) |
| 2B | ~2 billion | ~4 GB | ~8 GB | ⚠️ Tight (FP16 only) |
| 3B | ~3 billion | ~6 GB | ~12 GB | ⚠️ Very tight (FP16, small batches) |
| 7B | ~7 billion | ~14 GB | ~28 GB | ❌ No (too large) |

**Note**: With LoRA finetuning, you need additional memory for:
- Gradients: ~2x model size
- Optimizer states: ~2x model size
- Training data: Variable

**Recommendation**: For 8GB RAM, stick to **1B-2B models** with FP16 and batch_size=1.

## Recommended Models

### 🥇 **Best Options (1B-2B)**

#### 1. **Gemma 2B / Gemma 3 1B**
- **Hugging Face**: `google/gemma-2b-it` or `google/gemma-3-1b-it`
- **Size**: 1B-2B parameters
- **Memory**: ~2-4 GB (FP16)
- **Text-to-SQL**: Good performance, can be finetuned
- **Ollama**: Available as `gemma:2b` or `gemma:1b`
- **Pros**: 
  - Excellent for 8GB RAM
  - Good instruction-following capabilities
  - Well-documented
- **Cons**: 
  - May need more finetuning data than specialized models
- **Config Example**:
  ```yaml
  model:
    name: "google/gemma-2b-it"
    ollama_name: "gemma:2b"
  ```

#### 2. **Gemma 3 1B Text-to-SQL (Pre-finetuned)**
- **Hugging Face**: `Yuk050/gemma-3-1b-text-to-sql-model`
- **Size**: 1B parameters
- **Memory**: ~2 GB (FP16)
- **Text-to-SQL**: Already finetuned for text-to-SQL!
- **Pros**: 
  - Already optimized for text-to-SQL
  - Smallest viable option
  - Can be further finetuned
- **Cons**: 
  - May have lower baseline performance than larger models
- **Config Example**:
  ```yaml
  model:
    name: "Yuk050/gemma-3-1b-text-to-sql-model"
    ollama_name: "gemma-sql:1b"
  ```

#### 3. **T5-Small Awesome Text-to-SQL**
- **Hugging Face**: `cssupport/t5-small-awesome-text-to-sql`
- **Size**: ~60M parameters (very small!)
- **Memory**: ~120 MB (FP32)
- **Text-to-SQL**: Specialized for text-to-SQL
- **Pros**: 
  - Extremely lightweight
  - Fast inference
  - Already finetuned for SQL
- **Cons**: 
  - Encoder-decoder architecture (different from LLMs)
  - May need code modifications
  - Lower performance ceiling
- **Note**: Uses T5 architecture, may require different finetuning approach

#### 4. **Phi-2 / Phi-3 Mini**
- **Hugging Face**: `microsoft/phi-2` or `microsoft/Phi-3-mini-4k-instruct`
- **Size**: 2.7B-3.8B parameters
- **Memory**: ~5-7 GB (FP16)
- **Text-to-SQL**: Good general capabilities
- **Pros**: 
  - Strong performance for size
  - Good instruction following
- **Cons**: 
  - Pushing limits of 8GB RAM
  - May need very small batches
- **Config Example**:
  ```yaml
  model:
    name: "microsoft/Phi-3-mini-4k-instruct"
    ollama_name: "phi3:mini"
  ```

### 🥈 **Alternative Options (2B-3B)**

#### 5. **Mistral 7B Instruct (Quantized via Ollama)**
- **Ollama**: `mistral:7b-instruct-q4_0`
- **Size**: 7B (but quantized to ~4GB)
- **Memory**: ~4-5 GB (quantized)
- **Text-to-SQL**: Excellent performance
- **Pros**: 
  - Best performance in this list
  - Available via Ollama (handles quantization)
  - Can use for inference
- **Cons**: 
  - Cannot finetune quantized version locally
  - Would need full model for finetuning (too large)
- **Recommendation**: Use for inference/evaluation only, not finetuning

#### 6. **TinyLlama**
- **Hugging Face**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size**: 1.1B parameters
- **Memory**: ~2.2 GB (FP16)
- **Text-to-SQL**: Decent for size
- **Pros**: 
  - Very lightweight
  - Fast training
  - Good for experimentation
- **Cons**: 
  - Lower performance than Gemma
  - May need more finetuning

## Configuration Examples

### Example 1: Gemma 2B (Recommended)

Create `config-gemma2b.yaml`:

```yaml
model:
  name: "google/gemma-2b-it"
  ollama_name: "gemma:2b"
  max_tokens: 512
  temperature: 0.1
  top_p: 0.9

benchmark:
  name: "spider"
  test_split: "test"
  database_path: "./data/benchmarks/spider/database"
  data_path: "./data/benchmarks/spider"

evaluation:
  metrics:
    - "exact_match"
    - "execution_accuracy"
  timeout: 30
  max_examples: null

finetuning:
  method: "lora"  # LoRA only (no QLoRA on macOS)
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  learning_rate: 2e-4
  batch_size: 1  # Critical for 8GB RAM
  num_epochs: 3
  gradient_accumulation_steps: 8
  max_seq_length: 1024
  load_in_4bit: false
  load_in_8bit: false
  output_dir: "./models/finetuned-gemma2b"
  save_steps: 500
  eval_steps: 500

ollama:
  base_url: "http://localhost:11434"
  timeout: 300
  context_window: 4096

paths:
  models_dir: "./models"
  data_dir: "./data"
  results_dir: "./results"

logging:
  level: "INFO"
  use_wandb: false
  wandb_project: "text-to-sql-finetuning"
```

### Example 2: Gemma 3 1B Text-to-SQL (Pre-finetuned)

Create `config-gemma1b-sql.yaml`:

```yaml
model:
  name: "Yuk050/gemma-3-1b-text-to-sql-model"
  ollama_name: "gemma-sql:1b"
  max_tokens: 512
  temperature: 0.1
  top_p: 0.9

# ... rest same as above ...

finetuning:
  method: "lora"
  lora_r: 4  # Even smaller for 1B model
  lora_alpha: 8
  lora_dropout: 0.1
  learning_rate: 1e-4  # Lower LR for already-finetuned model
  batch_size: 2  # Can use slightly larger batch with 1B model
  num_epochs: 2  # Fewer epochs needed
  gradient_accumulation_steps: 4
  max_seq_length: 1024
  load_in_4bit: false
  load_in_8bit: false
  output_dir: "./models/finetuned-gemma1b-sql"
  save_steps: 500
  eval_steps: 500
```

## Memory Optimization Tips

### 1. Use FP16 Precision
```python
# In finetuner.py, ensure:
model_kwargs['torch_dtype'] = torch.float16  # Instead of float32
```

### 2. Gradient Checkpointing
```python
# Add to TrainingArguments:
gradient_checkpointing=True
```

### 3. Small Batch Sizes
```yaml
finetuning:
  batch_size: 1  # Absolute minimum
  gradient_accumulation_steps: 8  # Maintains effective batch size
```

### 4. Reduce Sequence Length
```yaml
finetuning:
  max_seq_length: 512  # Instead of 2048
```

### 5. Smaller LoRA Rank
```yaml
finetuning:
  lora_r: 4  # Instead of 16
  lora_alpha: 8  # Instead of 32
```

## Quick Start with Small Models

### Option A: Use Pre-finetuned Model (Easiest)

```bash
# 1. Use Gemma 3 1B Text-to-SQL
# Update config.yaml:
model:
  name: "Yuk050/gemma-3-1b-text-to-sql-model"

# 2. Run evaluation
python scripts/evaluate_baseline.py --config config-gemma1b-sql.yaml

# 3. Further finetune if needed
python scripts/finetune_model.py --config config-gemma1b-sql.yaml
```

### Option B: Finetune from Scratch

```bash
# 1. Use Gemma 2B
# Update config.yaml:
model:
  name: "google/gemma-2b-it"

# 2. Finetune
python scripts/finetune_model.py --config config-gemma2b.yaml
```

## Model Comparison Table

| Model | Size | Memory | Baseline SQL | Finetuning | Ollama | Best For |
|-------|------|--------|--------------|------------|--------|----------|
| Gemma 3 1B SQL | 1B | ~2GB | ⭐⭐⭐⭐ | ✅ Easy | ✅ Yes | Quick start |
| Gemma 2B | 2B | ~4GB | ⭐⭐⭐ | ✅ Easy | ✅ Yes | Balanced |
| Phi-3 Mini | 3.8B | ~7GB | ⭐⭐⭐⭐ | ⚠️ Tight | ✅ Yes | Best performance |
| T5-Small SQL | 60M | ~120MB | ⭐⭐⭐ | ⚠️ Different arch | ❌ No | Lightweight |
| TinyLlama | 1.1B | ~2GB | ⭐⭐ | ✅ Easy | ✅ Yes | Experimentation |

## Recommendations by Use Case

### 🎯 **Best Overall: Gemma 2B**
- Good balance of performance and memory
- Well-supported and documented
- Works well with LoRA finetuning

### 🚀 **Quickest Start: Gemma 3 1B Text-to-SQL**
- Already finetuned for SQL
- Smallest viable option
- Can be further improved

### 💪 **Best Performance: Phi-3 Mini**
- Strongest capabilities
- Pushes 8GB RAM limits
- Requires careful memory management

### 🔬 **Experimentation: TinyLlama**
- Fastest training
- Good for learning
- Lower performance ceiling

## Testing Model Compatibility

Before committing to finetuning, test if model loads:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2b-it"

# Check memory before loading
print(f"Available RAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" 
      if torch.cuda.is_available() else "CPU mode")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in FP16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto" if torch.cuda.is_available() else None
)

# Check memory after loading
if torch.cuda.is_available():
    print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
else:
    import psutil
    print(f"RAM Used: {psutil.virtual_memory().used / 1e9:.2f} GB")

print("Model loaded successfully!")
```

## Next Steps

1. **Choose a model** from the recommendations above
2. **Create config file** (use examples provided)
3. **Test model loading** (use script above)
4. **Start with evaluation** to establish baseline
5. **Finetune** with LoRA (not QLoRA)
6. **Compare results** with baseline

## Additional Resources

- [Gemma Models on Hugging Face](https://huggingface.co/google/gemma-2b-it)
- [Phi-3 Models](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Ollama Model Library](https://ollama.com/library)

---

**Remember**: With 8GB RAM, smaller models (1B-2B) are your best bet. Focus on LoRA finetuning with FP16 precision and batch_size=1.

