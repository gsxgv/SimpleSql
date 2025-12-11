"""
Model finetuner using LoRA/QLoRA for text-to-SQL tasks.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import json

logger = logging.getLogger(__name__)


class TextToSQLFinetuner:
    """Finetune models for text-to-SQL tasks using LoRA/QLoRA."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize finetuner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['model']['name']
        self.finetuning_config = config['finetuning']
        self.method = self.finetuning_config['method'].lower()
        self.models_dir = Path(config['paths']['models_dir'])
        self.output_dir = Path(self.finetuning_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.learning_rate = self.finetuning_config['learning_rate']
        self.batch_size = self.finetuning_config['batch_size']
        self.num_epochs = self.finetuning_config['num_epochs']
        self.gradient_accumulation_steps = self.finetuning_config['gradient_accumulation_steps']
        self.max_seq_length = self.finetuning_config['max_seq_length']
        self.save_steps = self.finetuning_config.get('save_steps', 500)
        self.eval_steps = self.finetuning_config.get('eval_steps', 500)
        
        # LoRA parameters
        self.lora_r = self.finetuning_config['lora_r']
        self.lora_alpha = self.finetuning_config['lora_alpha']
        self.lora_dropout = self.finetuning_config['lora_dropout']
        
        # Quantization
        self.load_in_4bit = self.finetuning_config.get('load_in_4bit', False)
        self.load_in_8bit = self.finetuning_config.get('load_in_8bit', False)
        
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check for Apple Silicon (MPS) or CUDA
        use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        use_cuda = torch.cuda.is_available()
        
        # Load model with quantization if specified
        model_kwargs = {
            'trust_remote_code': True,
            'device_map': 'auto' if (use_cuda or use_mps) else None,
        }
        
        # Check if bitsandbytes is available (only works on CUDA)
        bitsandbytes_available = False
        if use_cuda and self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bitsandbytes_available = True
            except ImportError:
                logger.warning("bitsandbytes not available. QLoRA requires CUDA. Using LoRA instead.")
                self.load_in_4bit = False
                self.load_in_8bit = False
        
        if self.load_in_4bit and bitsandbytes_available:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Loading model in 4-bit quantization")
        elif self.load_in_8bit and bitsandbytes_available:
            model_kwargs['load_in_8bit'] = True
            logger.info("Loading model in 8-bit quantization")
        elif (self.load_in_4bit or self.load_in_8bit) and not bitsandbytes_available:
            logger.warning(
                "Quantization requested but bitsandbytes not available. "
                "This is expected on Apple Silicon. Using FP16/FP32 instead."
            )
            # Use FP16 on MPS, FP32 on CPU
            if use_mps:
                model_kwargs['torch_dtype'] = torch.float16
                logger.info("Using FP16 on Apple Silicon (MPS)")
            else:
                model_kwargs['torch_dtype'] = torch.float32
                logger.info("Using FP32 on CPU")
        
        # Determine dtype based on available hardware
        use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        use_cuda = torch.cuda.is_available()
        
        if 'torch_dtype' not in model_kwargs:
            if use_cuda or use_mps:
                model_kwargs['torch_dtype'] = torch.float16
            else:
                model_kwargs['torch_dtype'] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to appropriate device
        if use_mps:
            self.model = self.model.to('mps')
            logger.info("Model moved to MPS (Apple Silicon GPU)")
        elif use_cuda:
            logger.info("Model on CUDA")
        else:
            logger.info("Model on CPU")
        
        logger.info("Model and tokenizer loaded successfully")
    
    def format_instruction(self, question: str, sql: str, schema: Optional[str] = None) -> str:
        """
        Format training example as instruction-following prompt.
        
        Args:
            question: Natural language question
            sql: SQL query
            schema: Optional database schema
            
        Returns:
            Formatted instruction string
        """
        parts = []
        
        if schema:
            parts.append(f"Database Schema:\n{schema}\n")
        
        parts.append(f"Question: {question}")
        parts.append(f"SQL: {sql}")
        
        return "\n".join(parts)
    
    def prepare_spider_dataset(self, split: str = "train") -> Dataset:
        """
        Prepare Spider dataset for finetuning.
        
        Args:
            split: Dataset split (train, dev)
            
        Returns:
            Hugging Face Dataset
        """
        data_path = Path(self.config['benchmark']['data_path'])
        data_file = data_path / f"{split}.json"
        schema_file = data_path / "tables.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Spider dataset file not found: {data_file}")
        
        # Load data
        with open(data_file, 'r') as f:
            examples = json.load(f)
        
        # Load schemas
        schemas = {}
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                tables = json.load(f)
                for table_info in tables:
                    db_id = table_info['db_id']
                    schema_parts = [f"Database: {db_id}"]
                    for table in table_info.get('table_names_original', []):
                        schema_parts.append(f"\nTable: {table}")
                        if 'column_names_original' in table_info:
                            cols = [
                                col[1] for col in table_info['column_names_original']
                                if col[0] == table_info['table_names_original'].index(table)
                            ]
                            schema_parts.append(f"  Columns: {', '.join(cols)}")
                    schemas[db_id] = "\n".join(schema_parts)
        
        # Format examples
        formatted_examples = []
        for ex in examples:
            question = ex['question']
            sql = ex['sql']
            db_id = ex['db_id']
            schema = schemas.get(db_id, f"Database: {db_id}")
            
            instruction = self.format_instruction(question, sql, schema)
            formatted_examples.append({'text': instruction})
        
        return Dataset.from_list(formatted_examples)
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length'
        )
    
    def setup_lora(self):
        """Setup LoRA/QLoRA configuration."""
        if self.method not in ['lora', 'qlora']:
            raise ValueError(f"Unsupported finetuning method: {self.method}")
        
        # Prepare model for training if using QLoRA (only if bitsandbytes is available)
        if self.method == 'qlora' and (self.load_in_4bit or self.load_in_8bit):
            try:
                self.model = prepare_model_for_kbit_training(self.model)
            except Exception as e:
                logger.warning(
                    f"Could not prepare model for k-bit training: {e}. "
                    "This is expected on Apple Silicon. Falling back to LoRA."
                )
                self.method = 'lora'
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self._get_target_modules(),
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied")
    
    def _get_target_modules(self) -> List[str]:
        """
        Get target modules for LoRA based on model architecture.
        
        Returns:
            List of target module names
        """
        model_type = self.model.config.model_type.lower()
        
        if 'llama' in model_type or 'mistral' in model_type:
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif 'gpt' in model_type:
            return ['c_attn', 'c_proj']
        else:
            # Default: try common attention modules
            return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    def finetune(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Finetune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=100,
            logging_steps=50,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=torch.cuda.is_available() or (torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False),
            bf16=False,
            report_to="wandb" if self.config['logging'].get('use_wandb', False) else None,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        logger.info("Finetuning complete!")
    
    def export_for_ollama(self, output_path: Optional[Path] = None):
        """
        Export finetuned model for Ollama.
        Note: This is a placeholder - actual export may require conversion to GGUF format.
        
        Args:
            output_path: Path to save exported model
        """
        if output_path is None:
            output_path = self.output_dir / "ollama_export"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to {output_path}")
        
        # Merge LoRA weights back into base model
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        logger.info(
            "Model exported. Note: You may need to convert to GGUF format "
            "for Ollama using llama.cpp or similar tools."
        )

