# Quick Start Guide

## Prerequisites

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Step-by-Step Workflow

### 1. Download and Setup Model

```bash
# Use default config.yaml
python scripts/download_model.py

# Or specify a different config file
python scripts/download_model.py --config config-gemma2b.yaml
```

This will:
- Download the model from Hugging Face (configured in the specified config file)
- Attempt to import it to Ollama
- Note: Some models may need manual conversion to GGUF format

**Alternative**: If the model is already available in Ollama's library:
```bash
ollama pull sqlcoder-7b
```

### 2. Download Benchmark Dataset

For Spider benchmark:
1. Download from: https://yale-lily.github.io/spider
2. Extract to `data/benchmarks/spider/`
3. Set up MySQL databases (see Spider documentation)

Update `config.yaml` with correct paths:
```yaml
benchmark:
  database_path: "./data/benchmarks/spider/database"
  data_path: "./data/benchmarks/spider"
```

### 3. Run Baseline Evaluation

```bash
python scripts/evaluate_baseline.py --split test --max-examples 100
```

This will:
- Evaluate the baseline model on the test set
- Calculate exact match and execution accuracy
- Save results to `results/`

### 4. Finetune the Model

```bash
python scripts/finetune_model.py --train-split train --eval-split dev
```

This will:
- Load training data from Spider
- Finetune using QLoRA (configured in `config.yaml`)
- Save checkpoints to `models/finetuned/`
- Optionally export for Ollama

**Note**: Finetuning requires GPU. Adjust batch size and other parameters in `config.yaml` if needed.

### 5. Import Finetuned Model to Ollama

After finetuning, you'll need to:
1. Convert the finetuned model to GGUF format (using llama.cpp or similar)
2. Import to Ollama:
   ```bash
   ollama import <path_to_modelfile>
   ```

Or update `config.yaml` to use the finetuned model name if it's already imported.

### 6. Evaluate Finetuned Model

```bash
python scripts/evaluate_finetuned.py --split test --max-examples 100
```

This will:
- Evaluate the finetuned model
- Compare with baseline results
- Generate comparison report

### 7. Analyze Results

Open `notebooks/analysis.ipynb` to:
- Visualize performance improvements
- Analyze error patterns
- Compare baseline vs finetuned predictions

## Configuration

Edit `config.yaml` to customize:
- **Model**: Change `model.name` and `model.ollama_name`
- **Benchmark**: Change `benchmark.name` and paths
- **Finetuning**: Adjust hyperparameters in `finetuning` section
- **Evaluation**: Modify metrics and timeout settings

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list
```

### Model Not Found
- Check model name in `config.yaml`
- Verify model is imported: `ollama list`
- Try pulling from Ollama library: `ollama pull <model-name>`

### Database Connection Issues
- Ensure MySQL is running (for Spider)
- Check database paths in `config.yaml`
- Verify database files exist

### GPU Memory Issues
- Reduce `batch_size` in `config.yaml`
- Enable `load_in_4bit: true` for QLoRA
- Use gradient accumulation

## Next Steps

1. Experiment with different models
2. Try different finetuning hyperparameters
3. Test on other benchmarks (WikiSQL, BIRD)
4. Analyze results and iterate

For detailed information, see `PLAN.md` and `README.md`.

