# Required Libraries and Dependencies

## Python Libraries

### Core Model & Training
- **transformers** (>=4.35.0): Hugging Face transformers library for model loading and manipulation
- **torch** (>=2.0.0): PyTorch for deep learning operations
- **peft** (>=0.6.0): Parameter-Efficient Fine-Tuning (LoRA, QLoRA) from Hugging Face
- **accelerate** (>=0.24.0): Training acceleration and distributed training support
- **bitsandbytes** (>=0.41.0): Quantization library for 4-bit/8-bit model loading (required for QLoRA)

### Data Handling
- **datasets** (>=2.14.0): Hugging Face datasets library for loading benchmark datasets
- **pandas** (>=2.0.0): Data manipulation and analysis
- **numpy** (>=1.24.0): Numerical operations

### Database & SQL
- **sqlparse** (>=0.4.4): SQL parsing and formatting
- **sqlalchemy** (>=2.0.0): Database connectivity and SQL execution
- **pymysql** (>=1.1.0): MySQL connector (required for Spider benchmark databases)

### Evaluation
- **sacrebleu** (>=2.3.0): BLEU score calculation (optional, for text similarity metrics)
- **rouge-score** (>=0.1.2): ROUGE score calculation (optional, for text similarity metrics)

### Ollama Integration
- **requests** (>=2.31.0): HTTP library for Ollama REST API calls
- **ollama** (>=0.1.0): Official Ollama Python client (if available, otherwise use requests)

### Utilities
- **pyyaml** (>=6.0): YAML configuration file parsing
- **tqdm** (>=4.66.0): Progress bars for long-running operations
- **python-dotenv** (>=1.0.0): Environment variable management
- **wandb** (>=0.15.0): Weights & Biases for experiment tracking (optional but recommended)

### Additional Dependencies
- **sentencepiece** (>=0.1.99): Tokenizer support for certain models
- **protobuf** (>=4.24.0): Protocol buffer support for model serialization

## External Tools (Not Python Packages)

### Ollama
- **Installation**: 
  - macOS: `brew install ollama`
  - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
  - Windows: Download from [ollama.ai](https://ollama.ai)
- **Purpose**: Local model inference server
- **API**: REST API at `http://localhost:11434`
- **Documentation**: https://ollama.ai/docs

### llama.cpp (Optional)
- **Purpose**: Convert models to GGUF format for Ollama
- **When needed**: If model conversion is required
- **Alternative**: Use Ollama's built-in import functionality

### MySQL (For Spider Benchmark)
- **Purpose**: Required to run Spider benchmark databases
- **Installation**: 
  - macOS: `brew install mysql`
  - Linux: `sudo apt-get install mysql-server`
  - Or use Docker: `docker run -d -p 3306:3306 mysql:8.0`

## Installation Commands

### Full Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve
```

### GPU Support (Optional but Recommended)
For CUDA support with PyTorch:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Library Usage Summary

| Library | Purpose | Critical? |
|---------|---------|-----------|
| transformers | Model loading/manipulation | ✅ Yes |
| torch | Deep learning framework | ✅ Yes |
| peft | Parameter-efficient finetuning | ✅ Yes (for finetuning) |
| datasets | Benchmark data loading | ✅ Yes |
| sqlalchemy | Database operations | ✅ Yes |
| requests | Ollama API calls | ✅ Yes |
| pyyaml | Configuration | ✅ Yes |
| bitsandbytes | Quantization | ⚠️ Yes (for QLoRA) |
| accelerate | Training acceleration | ⚠️ Recommended |
| wandb | Experiment tracking | ❌ Optional |
| sacrebleu | BLEU scoring | ❌ Optional |

## Version Compatibility Notes

- **Python**: Requires Python 3.8+ (3.10+ recommended)
- **PyTorch**: Version 2.0+ required for latest transformers features
- **CUDA**: If using GPU, ensure CUDA version matches PyTorch build
- **bitsandbytes**: May require specific CUDA versions (check compatibility)

## Troubleshooting Common Issues

### bitsandbytes Installation
```bash
# If installation fails, try:
pip install bitsandbytes --no-cache-dir

# Or install from source:
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

### Ollama Connection
```bash
# Test Ollama API
curl http://localhost:11434/api/tags

# If connection fails, ensure Ollama is running:
ollama serve
```

### Database Connection
```bash
# Test MySQL connection
mysql -u root -p

# For Spider benchmark, ensure databases are set up:
# See benchmark-specific setup instructions
```

## Additional Resources

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Spider Benchmark Setup](https://yale-lily.github.io/spider)

