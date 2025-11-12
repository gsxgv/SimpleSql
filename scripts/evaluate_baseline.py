#!/usr/bin/env python3
"""
Evaluate baseline model on benchmark.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import SQLEvaluator
from src.ollama_manager import OllamaManager
from src.utils import load_config, setup_logging, ensure_dir

def main():
    """Run baseline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate baseline model")
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to evaluate on (default: test)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples to evaluate (default: all)'
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
    ensure_dir(config['paths']['results_dir'])
    
    # Create Ollama manager
    ollama_manager = OllamaManager(config)
    
    # Check connection
    if not ollama_manager.check_connection():
        print("Error: Ollama is not running or not accessible.")
        print("Please start Ollama: ollama serve")
        sys.exit(1)
    
    # Check model exists
    if not ollama_manager.check_model_exists():
        print(f"Error: Model {config['model']['ollama_name']} not found in Ollama.")
        print(f"Available models: {ollama_manager.list_models()}")
        print(f"Please download and setup the model first: python scripts/download_model.py")
        sys.exit(1)
    
    # Create evaluator
    evaluator = SQLEvaluator(config, ollama_manager)
    
    # Run evaluation
    print(f"Evaluating baseline model on {config['benchmark']['name']} {args.split} split...")
    results = evaluator.evaluate(
        split=args.split,
        max_examples=args.max_examples or config['evaluation'].get('max_examples')
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
    print(f"\nResults saved to: {results['output_file']}")
    print("="*50)

if __name__ == "__main__":
    main()

