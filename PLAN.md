# Text-to-SQL Model Finetuning Project Plan

## Overview
This project aims to finetune an open-source Hugging Face model for improved text-to-SQL functionality. The model will be run locally via Ollama, evaluated on industry benchmarks, finetuned, and re-evaluated to measure improvement.

## Project Structure

```
simplesql/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── model_downloader.py      # Download and prepare Hugging Face models
│   ├── ollama_manager.py         # Manage Ollama model operations
│   ├── evaluator.py              # Benchmark evaluation framework
│   ├── finetuner.py              # Model finetuning logic
│   └── utils.py                  # Utility functions
├── data/
│   ├── benchmarks/               # Benchmark datasets
│   └── finetuning/               # Finetuning datasets
├── models/                       # Local model storage
├── results/                      # Evaluation results
├── scripts/
│   ├── download_model.py
│   ├── evaluate_baseline.py
│   ├── finetune_model.py
│   └── evaluate_finetuned.py
└── notebooks/
    └── analysis.ipynb            # Results analysis
```

## Components

### 1. Model Selection & Download
**Target Models** (suitable for text-to-SQL):
- `defog/sqlcoder-7b-2` (specialized text-to-SQL model)
- `mistralai/Mistral-7B-Instruct-v0.2` (general purpose, good for SQL)
- `meta-llama/Llama-2-7b-chat-hf` (if available)
- `NumbersStation/nsql-llama-2-7B` (SQL-specific)

**Implementation**:
- Download model from Hugging Face using `transformers` library
- Convert to Ollama-compatible format (GGUF format)
- Store locally in `models/` directory

### 2. Ollama Integration
**Requirements**:
- Ollama installed and running locally
- Model converted to GGUF format for Ollama
- API client for Ollama REST API

**Functionality**:
- Import model into Ollama
- Generate SQL queries from natural language prompts
- Handle model inference via Ollama API

### 3. Benchmark Evaluation
**Industry Benchmarks**:
- **Spider** (most popular): Complex cross-domain text-to-SQL dataset
- **WikiSQL**: Large-scale dataset with simple SQL queries
- **BIRD**: Includes realistic database values and external knowledge
- **CSpider**: Chinese version of Spider

**Evaluation Metrics**:
- **Exact Match (EM)**: Percentage of queries that match exactly
- **Execution Accuracy**: Percentage of queries that execute correctly and return correct results
- **Component-level accuracy**: Accuracy of SELECT, WHERE, JOIN clauses separately

**Implementation**:
- Load benchmark datasets
- Generate SQL queries using the model
- Execute queries against provided databases
- Compare results and calculate metrics

### 4. Finetuning Framework
**Approach Options**:
- **LoRA/QLoRA**: Parameter-efficient finetuning (recommended)
- **Full finetuning**: Complete model finetuning (requires more resources)
- **PEFT**: Hugging Face Parameter-Efficient Fine-Tuning library

**Text-to-SQL Libraries**:
- `sqlcoder`: Specialized for SQL generation
- `text2sql`: General text-to-SQL framework
- Custom finetuning using `transformers` + `peft` + `datasets`

**Finetuning Data**:
- Use Spider training set or similar
- Format: (natural language question, SQL query, database schema) pairs
- Data augmentation techniques

### 5. Evaluation Pipeline
**Pre-finetuning**:
- Run baseline evaluation on test set
- Store results and predictions
- Generate evaluation report

**Post-finetuning**:
- Run evaluation on same test set
- Compare with baseline
- Generate comparison report

## Required Libraries

### Core Dependencies
```python
# Model & Transformers
transformers>=4.35.0          # Hugging Face transformers
torch>=2.0.0                  # PyTorch
peft>=0.6.0                   # Parameter-efficient finetuning
accelerate>=0.24.0            # Training acceleration
bitsandbytes>=0.41.0          # Quantization (for QLoRA)

# Data Handling
datasets>=2.14.0              # Hugging Face datasets
pandas>=2.0.0                 # Data manipulation
numpy>=1.24.0                 # Numerical operations

# Database & SQL
sqlparse>=0.4.4               # SQL parsing
sqlalchemy>=2.0.0             # Database connectivity
pymysql>=1.1.0                # MySQL connector (for Spider)
sqlite3                      # Built-in SQLite support

# Evaluation
sacrebleu>=2.3.0              # BLEU scoring (optional)
rouge-score>=0.1.2            # ROUGE scoring (optional)

# Ollama Integration
requests>=2.31.0              # HTTP requests for Ollama API
ollama>=0.1.0                 # Ollama Python client (if available)

# Utilities
pyyaml>=6.0                   # Configuration files
tqdm>=4.66.0                  # Progress bars
python-dotenv>=1.0.0          # Environment variables
wandb>=0.15.0                 # Experiment tracking (optional)
```

### Additional Tools
- **Ollama**: Must be installed separately (not a Python package)
- **llama.cpp**: For GGUF conversion (if needed)
- **sentencepiece**: Tokenizer support
- **protobuf**: Model serialization

## Implementation Steps

### Phase 1: Setup & Infrastructure
1. Set up project structure
2. Create `requirements.txt` with all dependencies
3. Create configuration file (`config.yaml`) for:
   - Model selection
   - Benchmark selection
   - Ollama settings
   - Finetuning hyperparameters
4. Install Ollama and verify it's running

### Phase 2: Model Download & Ollama Integration
1. Implement `model_downloader.py`:
   - Download model from Hugging Face
   - Convert to GGUF format (if needed)
   - Import to Ollama
2. Implement `ollama_manager.py`:
   - Model inference wrapper
   - Prompt formatting for text-to-SQL
   - Error handling

### Phase 3: Baseline Evaluation
1. Download benchmark dataset (Spider recommended)
2. Implement `evaluator.py`:
   - Load benchmark data
   - Generate SQL queries via Ollama
   - Execute queries against test databases
   - Calculate accuracy metrics
3. Run baseline evaluation
4. Store results

### Phase 4: Finetuning
1. Prepare finetuning dataset:
   - Load training data from benchmark
   - Format as instruction-following dataset
   - Create data loaders
2. Implement `finetuner.py`:
   - Setup LoRA/QLoRA configuration
   - Training loop
   - Model checkpointing
   - Convert finetuned model to Ollama format
3. Run finetuning
4. Export finetuned model to Ollama

### Phase 5: Post-Finetuning Evaluation
1. Run evaluation on finetuned model
2. Compare with baseline results
3. Generate comparison report
4. Analyze improvements

## Configuration Example

```yaml
# config.yaml
model:
  name: "defog/sqlcoder-7b-2"
  ollama_name: "sqlcoder-7b"
  max_tokens: 512
  temperature: 0.1

benchmark:
  name: "spider"
  test_split: "test"
  database_path: "./data/benchmarks/spider/database"

evaluation:
  metrics: ["exact_match", "execution_accuracy"]
  timeout: 30  # seconds per query

finetuning:
  method: "qlora"
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  learning_rate: 2e-4
  batch_size: 4
  num_epochs: 3
  gradient_accumulation_steps: 4
  max_seq_length: 2048

ollama:
  base_url: "http://localhost:11434"
  timeout: 300
```

## Key Considerations

1. **Model Format**: Ollama uses GGUF format. May need conversion tools like `llama.cpp` or `llama-cpp-python`
2. **Database Setup**: Spider benchmark requires MySQL databases. Need to set up and populate test databases
3. **Resource Requirements**: 
   - 7B models need ~14GB RAM (FP16) or ~7GB (quantized)
   - Finetuning requires GPU (recommended) or significant CPU time
4. **Evaluation Complexity**: Execution accuracy requires actual database execution, which can be slow
5. **Prompt Engineering**: Text-to-SQL performance heavily depends on prompt format (schema, examples, etc.)

## Success Metrics

- Baseline exact match accuracy: Target >40% on Spider
- Post-finetuning improvement: Target +10-20% absolute improvement
- Execution accuracy: Target >50% on Spider
- Training time: <24 hours on single GPU for 7B model

## Next Steps

1. Review and approve this plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Iterate based on results

