#!/usr/bin/env python3
"""
Evaluate finetuned model and compare with baseline.
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import SQLEvaluator
from src.ollama_manager import OllamaManager
from src.utils import load_config, setup_logging, ensure_dir

def load_baseline_results(results_dir: Path) -> dict:
    """
    Load baseline evaluation results.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        Baseline results dictionary
    """
    # Find most recent baseline result
    baseline_files = list(results_dir.glob("evaluation_*_baseline.json"))
    if not baseline_files:
        # Try any evaluation file
        baseline_files = sorted(results_dir.glob("evaluation_*.json"))
    
    if not baseline_files:
        return None
    
    # Get most recent
    baseline_file = sorted(baseline_files, key=lambda x: x.stat().st_mtime)[-1]
    
    with open(baseline_file, 'r') as f:
        return json.load(f)

def compare_results(baseline: dict, finetuned: dict) -> dict:
    """
    Compare baseline and finetuned results.
    
    Args:
        baseline: Baseline results dictionary
        finetuned: Finetuned results dictionary
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        'baseline_metrics': baseline.get('metrics', {}),
        'finetuned_metrics': finetuned.get('metrics', {}),
        'improvements': {}
    }
    
    # Calculate improvements
    for metric in baseline.get('metrics', {}).keys():
        baseline_val = baseline['metrics'].get(metric, 0)
        finetuned_val = finetuned['metrics'].get(metric, 0)
        improvement = finetuned_val - baseline_val
        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
        
        comparison['improvements'][metric] = {
            'absolute': improvement,
            'relative_percent': improvement_pct
        }
    
    return comparison

def main():
    """Run finetuned model evaluation and comparison."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned model")
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
        '--baseline-results',
        type=str,
        default=None,
        help='Path to baseline results file (default: auto-detect)'
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
        sys.exit(1)
    
    # Create evaluator
    evaluator = SQLEvaluator(config, ollama_manager)
    
    # Run evaluation
    print(f"Evaluating finetuned model on {config['benchmark']['name']} {args.split} split...")
    finetuned_results = evaluator.evaluate(
        split=args.split,
        max_examples=args.max_examples or config['evaluation'].get('max_examples')
    )
    
    # Load baseline results
    results_dir = Path(config['paths']['results_dir'])
    if args.baseline_results:
        baseline_path = Path(args.baseline_results)
    else:
        baseline_path = None
    
    if baseline_path and baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
    else:
        baseline_results = load_baseline_results(results_dir)
    
    # Print results
    print("\n" + "="*50)
    print("FINETUNED MODEL EVALUATION RESULTS")
    print("="*50)
    for metric, value in finetuned_results['metrics'].items():
        print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
    
    # Compare with baseline if available
    if baseline_results:
        print("\n" + "="*50)
        print("COMPARISON WITH BASELINE")
        print("="*50)
        
        comparison = compare_results(baseline_results, finetuned_results)
        
        print("\nBaseline Metrics:")
        for metric, value in comparison['baseline_metrics'].items():
            print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
        
        print("\nFinetuned Metrics:")
        for metric, value in comparison['finetuned_metrics'].items():
            print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
        
        print("\nImprovements:")
        for metric, improvement in comparison['improvements'].items():
            abs_imp = improvement['absolute']
            rel_imp = improvement['relative_percent']
            sign = "+" if abs_imp >= 0 else ""
            print(f"  {metric}: {sign}{abs_imp:.4f} ({sign}{rel_imp:.2f}%)")
        
        # Save comparison
        comparison_file = results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump({
                'comparison': comparison,
                'baseline_file': str(baseline_path) if baseline_path else 'auto-detected',
                'finetuned_file': finetuned_results['output_file']
            }, f, indent=2)
        
        print(f"\nComparison saved to: {comparison_file}")
    else:
        print("\nNo baseline results found for comparison.")
        print("Run baseline evaluation first: python scripts/evaluate_baseline.py")
    
    print(f"\nResults saved to: {finetuned_results['output_file']}")
    print("="*50)

if __name__ == "__main__":
    main()

