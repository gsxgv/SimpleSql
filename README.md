# SimpleSQL: Text-to-SQL Model Finetuning Project

A Python project for finetuning open-source Hugging Face models to improve text-to-SQL functionality using Ollama for local inference and industry-standard benchmarks for evaluation.

## Project Overview

This project provides a complete pipeline for:
1. **Downloading** Hugging Face models and running them locally via Ollama
2. **Evaluating** models on text-to-SQL benchmarks (Spider, WikiSQL, BIRD)
3. **Finetuning** models using parameter-efficient techniques (LoRA/QLoRA)
4. **Re-evaluating** finetuned models to measure improvement

## Features

- 🚀 Local model inference via Ollama
- 📊 Industry-standard benchmark evaluation (Spider, WikiSQL, BIRD)
- 🎯 Parameter-efficient finetuning (LoRA/QLoRA)
- 📈 Comprehensive evaluation metrics (exact match, execution accuracy)
- 🔧 Configurable via YAML configuration file

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for finetuning)
- At least 16GB RAM (for 7B models)
- Ollama installed and running locally

### Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd simplesql
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama (if not already installed) and start the service:
```bash
ollama serve
```

## Project Structure

```
simplesql/
├── README.md                 # This file
├── PLAN.md                   # Detailed project plan
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── src/                      # Source code
│   ├── model_downloader.py   # Model download & Ollama setup
│   ├── ollama_manager.py     # Ollama API wrapper
│   ├── evaluator.py          # Benchmark evaluation
│   ├── finetuner.py          # Model finetuning
│   └── utils.py              # Utility functions
├── data/                     # Data directory
│   ├── benchmarks/           # Benchmark datasets
│   └── finetuning/           # Finetuning datasets
├── models/                   # Model storage
├── results/                  # Evaluation results
└── scripts/                  # Execution scripts
```

## Quick Start

### 1. Download and Setup Model

```bash
python scripts/download_model.py
```

This will:
- Download the model from Hugging Face
- Convert it to Ollama format (if needed)
- Import it into Ollama

### 2. Run Baseline Evaluation

```bash
python scripts/evaluate_baseline.py
```

This evaluates the model on the benchmark and saves results to `results/baseline/`.

### 3. Finetune the Model

```bash
python scripts/finetune_model.py
```

This will:
- Load the training dataset
- Finetune using LoRA/QLoRA
- Save checkpoints
- Export finetuned model to Ollama

### 4. Evaluate Finetuned Model

```bash
python scripts/evaluate_finetuned.py
```

This evaluates the finetuned model and compares with baseline results.

## Configuration

Edit `config.yaml` to customize:
- Model selection
- Benchmark dataset
- Finetuning hyperparameters
- Evaluation settings
- Ollama configuration

See `PLAN.md` for detailed configuration options.

## Benchmarks

### Spider
- **Description**: Complex cross-domain text-to-SQL dataset
- **Size**: ~10,000 questions across 200 databases
- **Difficulty**: Medium to Hard
- **Download**: [Spider Dataset](https://yale-lily.github.io/spider)

### WikiSQL
- **Description**: Large-scale dataset with simple SQL queries
- **Size**: ~80,000 questions
- **Difficulty**: Easy
- **Download**: [WikiSQL Dataset](https://github.com/salesforce/WikiSQL)

### BIRD
- **Description**: Realistic database values and external knowledge
- **Size**: ~12,000 questions
- **Difficulty**: Hard
- **Download**: [BIRD Dataset](https://bird-bench.github.io/)

## Evaluation Metrics

- **Exact Match (EM)**: Percentage of generated SQL queries that exactly match the ground truth
- **Execution Accuracy**: Percentage of queries that execute correctly and return correct results
- **Component Accuracy**: Accuracy of individual SQL components (SELECT, WHERE, JOIN, etc.)

## Model Recommendations

### For Text-to-SQL:
1. **defog/sqlcoder-7b-2** - Specialized for SQL generation
2. **mistralai/Mistral-7B-Instruct-v0.2** - Strong general-purpose model
3. **NumbersStation/nsql-llama-2-7B** - SQL-specific variant

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check Ollama API: `curl http://localhost:11434/api/tags`
- Verify model is imported: `ollama list`

### GPU Memory Issues
- Use QLoRA with 4-bit quantization
- Reduce batch size in `config.yaml`
- Use gradient accumulation

### Database Connection Issues
- Ensure MySQL is running (for Spider benchmark)
- Check database paths in `config.yaml`
- Verify database files are downloaded

## Contributing

Contributions are welcome! Please see `PLAN.md` for implementation details.

## License

[Specify your license here]

## References

- [Spider Benchmark](https://yale-lily.github.io/spider)
- [Ollama Documentation](https://ollama.ai/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)

