# Running SimpleSQL in Google Colab

This guide explains how to run the text-to-SQL finetuning project in Google Colab, which provides free GPU access perfect for finetuning models.

## Quick Reference

**Fastest Way to Get Started:**
1. Open [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Upload `notebooks/colab_finetuning.ipynb` or copy the template below
4. Run all cells

**Key Advantages of Colab:**
- ✅ Free GPU (T4/V100/A100)
- ✅ Full CUDA support (bitsandbytes works!)
- ✅ QLoRA with 4-bit quantization
- ✅ Can finetune 7B+ models easily

**Files You'll Need:**
- `notebooks/colab_finetuning.ipynb` - Ready-to-use notebook
- `config-colab.yaml` - Colab-optimized configuration
- Project source code (`src/` directory)

## Why Google Colab?

- ✅ **Free GPU access** (T4, V100, A100 available)
- ✅ **No hardware limitations** - Can finetune 7B+ models
- ✅ **Full CUDA support** - bitsandbytes and QLoRA work perfectly
- ✅ **Pre-installed libraries** - Many dependencies already available
- ✅ **Easy sharing** - Share notebooks with others

## Prerequisites

1. Google account
2. Access to Google Colab (free tier available)
3. Optional: Google Drive for persistent storage

## Quick Start

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → New Notebook**
3. Or use this direct link: [New Colab Notebook](https://colab.research.google.com/)

### Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 is free)
3. Click **Save**

### Step 3: Clone and Setup Project

Run this in the first cell:

```python
# Clone the repository
!git clone https://github.com/yourusername/simplesql.git
# Or upload your project files manually

# Navigate to project directory
import os
os.chdir('simplesql')

# Install dependencies
!pip install -q -r requirements.txt

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Complete Colab Notebook Template

Here's a complete notebook you can use:

```python
# ============================================================================
# Cell 1: Setup and Installation
# ============================================================================

# Clone repository (or upload files manually)
!git clone https://github.com/yourusername/simplesql.git
import os
os.chdir('simplesql')

# Install dependencies
!pip install -q transformers torch peft accelerate bitsandbytes datasets \
    sqlparse sqlalchemy pymysql pyyaml tqdm python-dotenv wandb \
    sentencepiece protobuf sacrebleu rouge-score requests

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Cell 2: Mount Google Drive (Optional - for persistent storage)
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

# Create directories in Drive for persistent storage
!mkdir -p /content/drive/MyDrive/simplesql/{models,data,results}

# ============================================================================
# Cell 3: Download Benchmark Dataset
# ============================================================================

# Download Spider dataset
!mkdir -p data/benchmarks/spider
!wget -q https://yale-lily.github.io/spider/dataset/spider.zip -O /tmp/spider.zip
!unzip -q /tmp/spider.zip -d data/benchmarks/spider/
!rm /tmp/spider.zip

# Verify download
import json
with open('data/benchmarks/spider/train_spider.json', 'r') as f:
    train_data = json.load(f)
print(f"Downloaded {len(train_data)} training examples")

# ============================================================================
# Cell 4: Configure Project
# ============================================================================

import yaml

# Update config for Colab
config = {
    'model': {
        'name': 'defog/sqlcoder-7b-2',
        'ollama_name': 'sqlcoder-7b',
        'max_tokens': 512,
        'temperature': 0.1,
        'top_p': 0.9
    },
    'benchmark': {
        'name': 'spider',
        'test_split': 'test',
        'database_path': './data/benchmarks/spider/database',
        'data_path': './data/benchmarks/spider'
    },
    'evaluation': {
        'metrics': ['exact_match', 'execution_accuracy'],
        'timeout': 30,
        'max_examples': None
    },
    'finetuning': {
        'method': 'qlora',  # Can use QLoRA in Colab!
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'learning_rate': 2e-4,
        'batch_size': 4,  # Can use larger batches with GPU
        'num_epochs': 3,
        'gradient_accumulation_steps': 4,
        'max_seq_length': 2048,
        'load_in_4bit': True,  # Works in Colab!
        'load_in_8bit': False,
        'output_dir': './models/finetuned',
        'save_steps': 500,
        'eval_steps': 500
    },
    'ollama': {
        'base_url': 'http://localhost:11434',
        'timeout': 300,
        'context_window': 4096
    },
    'paths': {
        'models_dir': './models',
        'data_dir': './data',
        'results_dir': './results'
    },
    'logging': {
        'level': 'INFO',
        'use_wandb': False,
        'wandb_project': 'text-to-sql-finetuning'
    }
}

# Save config
with open('config-colab.yaml', 'w') as f:
    yaml.dump(config, f)

print("Configuration saved to config-colab.yaml")

# ============================================================================
# Cell 5: Download Model (Optional - if not using Ollama)
# ============================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

model_name = "defog/sqlcoder-7b-2"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# ============================================================================
# Cell 6: Run Baseline Evaluation (if using Hugging Face directly)
# ============================================================================

# Note: For Ollama-based evaluation, you'd need to install Ollama in Colab
# See "Ollama in Colab" section below

from src.evaluator import SQLEvaluator
from src.ollama_manager import OllamaManager

# This would require Ollama setup - see below
# evaluator = SQLEvaluator(config, ollama_manager)
# results = evaluator.evaluate(split='dev', max_examples=100)

# ============================================================================
# Cell 7: Finetune Model
# ============================================================================

from src.finetuner import TextToSQLFinetuner
from src.utils import load_config

# Load config
config = load_config('config-colab.yaml')

# Create finetuner
finetuner = TextToSQLFinetuner(config)

# Prepare dataset
print("Preparing training dataset...")
train_dataset = finetuner.prepare_spider_dataset(split='train')
print(f"Loaded {len(train_dataset)} training examples")

# Prepare eval dataset
print("Preparing evaluation dataset...")
eval_dataset = finetuner.prepare_spider_dataset(split='dev')
print(f"Loaded {len(eval_dataset)} evaluation examples")

# Run finetuning
print("Starting finetuning...")
finetuner.finetune(train_dataset, eval_dataset)

print("Finetuning complete!")

# ============================================================================
# Cell 8: Save Model to Drive
# ============================================================================

# Copy finetuned model to Drive
!cp -r models/finetuned /content/drive/MyDrive/simplesql/models/

# Copy results
!cp -r results /content/drive/MyDrive/simplesql/

print("Model and results saved to Google Drive")

# ============================================================================
# Cell 9: Download Results
# ============================================================================

from google.colab import files
import zipfile

# Create zip of results
!zip -r results.zip results/
files.download('results.zip')

print("Results downloaded!")
```

## Using Ollama in Colab

Ollama can be installed in Colab, but it's more complex. Here's how:

```python
# Install Ollama in Colab
!curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama in background
import subprocess
import time

ollama_process = subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for Ollama to start
time.sleep(5)

# Verify Ollama is running
import requests
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    print("Ollama is running!" if response.status_code == 200 else "Ollama not responding")
except:
    print("Ollama not running")

# Import a model (this downloads the model)
!ollama pull sqlcoder-7b
```

**Note**: Ollama in Colab has limitations:
- Models are downloaded each session (unless saved to Drive)
- Requires background process management
- May be slower than direct Hugging Face inference

**Recommendation**: For Colab, use Hugging Face transformers directly instead of Ollama.

## Colab-Specific Considerations

### 1. Session Timeout

**Issue**: Colab sessions timeout after inactivity (90 minutes for free tier)

**Solutions**:
- Save checkpoints frequently
- Use Google Drive for persistent storage
- Download important files before session ends

```python
# Auto-save to Drive every N steps
# Add to TrainingArguments:
save_steps=100  # Save more frequently
```

### 2. GPU Memory Limits

**Free Tier**: ~15GB GPU memory (T4)
**Pro Tier**: Up to 80GB (A100)

**Tips**:
- Use 4-bit quantization (QLoRA) to fit larger models
- Reduce batch size if OOM errors occur
- Use gradient checkpointing

```python
# In config-colab.yaml:
finetuning:
  batch_size: 2  # Reduce if OOM
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### 3. Download Limits

**Issue**: Large model downloads may timeout

**Solutions**:
- Use `huggingface_hub` with resume capability
- Download to Drive first, then copy

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="defog/sqlcoder-7b-2",
    cache_dir="/content/drive/MyDrive/simplesql/models/cache"
)
```

### 4. Persistent Storage

**Use Google Drive** for:
- Model checkpoints
- Results
- Dataset cache

```python
# Mount Drive at start
from google.colab import drive
drive.mount('/content/drive')

# Update paths in config
config['paths']['models_dir'] = '/content/drive/MyDrive/simplesql/models'
config['paths']['results_dir'] = '/content/drive/MyDrive/simplesql/results'
```

## Complete Workflow Example

### Option A: Full Finetuning Pipeline

```python
# 1. Setup
!git clone https://github.com/yourusername/simplesql.git
os.chdir('simplesql')
!pip install -q -r requirements.txt

# 2. Download data
!wget https://yale-lily.github.io/spider/dataset/spider.zip
!unzip spider.zip -d data/benchmarks/spider/

# 3. Finetune
from src.finetuner import TextToSQLFinetuner
from src.utils import load_config

config = load_config('config-colab.yaml')
finetuner = TextToSQLFinetuner(config)
train_dataset = finetuner.prepare_spider_dataset('train')
finetuner.finetune(train_dataset)

# 4. Save to Drive
!cp -r models/finetuned /content/drive/MyDrive/simplesql/
```

### Option B: Evaluation Only

```python
# 1. Setup
!git clone https://github.com/yourusername/simplesql.git
os.chdir('simplesql')
!pip install -q -r requirements.txt

# 2. Load pre-finetuned model from Drive
# (assuming you finetuned earlier)

# 3. Evaluate
from src.evaluator import SQLEvaluator
# ... evaluation code
```

## Colab Configuration File

Create `config-colab.yaml` with these optimizations:

```yaml
finetuning:
  method: "qlora"  # Can use QLoRA in Colab!
  load_in_4bit: true  # Works with CUDA
  batch_size: 4  # Larger batches possible with GPU
  max_seq_length: 2048  # Can use longer sequences
  gradient_accumulation_steps: 4
  num_epochs: 3
  save_steps: 100  # Save more frequently for Colab
```

## Tips and Best Practices

### 1. Monitor GPU Usage

```python
# Check GPU memory
!nvidia-smi

# In code
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

### 2. Use WandB for Tracking

```python
# Enable WandB in config
config['logging']['use_wandb'] = True

# Login (first time)
import wandb
wandb.login()
```

### 3. Handle Interruptions

```python
# Save checkpoints frequently
# Resume from checkpoint if interrupted
trainer = Trainer(
    ...,
    resume_from_checkpoint='./models/finetuned/checkpoint-500'
)
```

### 4. Download Before Session Ends

```python
# Download important files
from google.colab import files

# Download results
files.download('results.zip')

# Download model (if small enough)
!zip -r finetuned_model.zip models/finetuned/
files.download('finetuned_model.zip')
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config['finetuning']['batch_size'] = 1

# Use gradient checkpointing
from transformers import Trainer
trainer = Trainer(
    ...,
    gradient_checkpointing=True
)
```

### Session Disconnected

- Checkpoints are saved to `./models/finetuned/`
- Copy to Drive immediately after training steps
- Resume from last checkpoint

### Slow Downloads

```python
# Use mirror or cache
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## Comparison: Colab vs Local (macOS)

| Feature | Colab | Local (M2, 8GB) |
|---------|-------|-----------------|
| GPU | ✅ Free T4/V100/A100 | ❌ No GPU |
| bitsandbytes | ✅ Works | ❌ Not available |
| QLoRA | ✅ Full support | ❌ Not available |
| 7B+ Models | ✅ Easy | ⚠️ Difficult |
| Batch Size | ✅ 4-8 | ⚠️ 1 only |
| Speed | ✅ Fast | ⚠️ Slow |
| Persistence | ⚠️ Need Drive | ✅ Local storage |
| Cost | ✅ Free (limited) | ✅ Free |

## Next Steps

1. **Open Colab**: [colab.research.google.com](https://colab.research.google.com/)
2. **Copy the notebook template** above
3. **Enable GPU** runtime
4. **Run cells sequentially**
5. **Save results to Drive** before session ends

## Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab GPU Limits](https://colab.research.google.com/signup)
- [Hugging Face Colab Examples](https://huggingface.co/docs/transformers/notebooks)

## Example: Complete Colab Notebook

See `notebooks/colab_example.ipynb` for a ready-to-use Colab notebook (if created).

---

**Pro Tip**: For best results, use Colab Pro ($10/month) for:
- Faster GPUs (V100, A100)
- Longer session times
- More memory
- Priority access

