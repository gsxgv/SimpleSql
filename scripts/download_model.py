#!/usr/bin/env python3
"""
Download and setup model for Ollama.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_downloader import ModelDownloader
from src.utils import load_config, setup_logging, ensure_dir

def main():
    """Download and setup model."""
    parser = argparse.ArgumentParser(description="Download and setup model for Ollama")
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
    
    # Create downloader
    downloader = ModelDownloader(config)
    
    # Download and setup model
    print(f"Downloading model: {config['model']['name']}")
    success = downloader.setup_model(force_download=False)
    
    if success:
        print("Model setup complete!")
        print(f"\nNext steps:")
        print(f"1. Verify model is in Ollama: ollama list")
        print(f"2. Test model: ollama run {config['model']['ollama_name']}")
        print(f"3. Run baseline evaluation: python scripts/evaluate_baseline.py")
    else:
        print("Model setup encountered issues. Please check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()

