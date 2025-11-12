#!/usr/bin/env python3
"""
Finetune model for text-to-SQL task.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetuner import TextToSQLFinetuner
from src.utils import load_config, setup_logging, ensure_dir

def main():
    """Run model finetuning."""
    parser = argparse.ArgumentParser(description="Finetune model for text-to-SQL")
    parser.add_argument(
        '--train-split',
        type=str,
        default='train',
        help='Training dataset split (default: train)'
    )
    parser.add_argument(
        '--eval-split',
        type=str,
        default=None,
        help='Evaluation dataset split (default: None, no evaluation)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config['logging']['level'])
    
    # Ensure directories exist
    ensure_dir(config['paths']['models_dir'])
    ensure_dir(config['finetuning']['output_dir'])
    
    # Initialize wandb if configured
    if config['logging'].get('use_wandb', False):
        import wandb
        wandb.init(
            project=config['logging'].get('wandb_project', 'text-to-sql-finetuning'),
            config=config
        )
    
    # Create finetuner
    finetuner = TextToSQLFinetuner(config)
    
    # Prepare datasets
    print(f"Preparing training dataset from {args.train_split} split...")
    train_dataset = finetuner.prepare_spider_dataset(split=args.train_split)
    print(f"Loaded {len(train_dataset)} training examples")
    
    eval_dataset = None
    if args.eval_split:
        print(f"Preparing evaluation dataset from {args.eval_split} split...")
        eval_dataset = finetuner.prepare_spider_dataset(split=args.eval_split)
        print(f"Loaded {len(eval_dataset)} evaluation examples")
    
    # Run finetuning
    print("\nStarting finetuning...")
    print(f"Method: {config['finetuning']['method']}")
    print(f"Epochs: {config['finetuning']['num_epochs']}")
    print(f"Batch size: {config['finetuning']['batch_size']}")
    print(f"Learning rate: {config['finetuning']['learning_rate']}")
    print(f"Output directory: {config['finetuning']['output_dir']}")
    
    finetuner.finetune(train_dataset, eval_dataset)
    
    # Export for Ollama (optional)
    export = input("\nExport model for Ollama? (y/n): ").lower().strip()
    if export == 'y':
        finetuner.export_for_ollama()
    
    print("\nFinetuning complete!")
    print(f"Model saved to: {config['finetuning']['output_dir']}")

if __name__ == "__main__":
    main()

