"""
Model downloader for Hugging Face models.
Handles downloading models and preparing them for Ollama.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download and prepare models from Hugging Face."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model downloader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['model']['name']
        self.ollama_name = config['model'].get('ollama_name', self.model_name.split('/')[-1])
        self.models_dir = Path(config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def download_model(self, force_download: bool = False) -> Path:
        """
        Download model from Hugging Face.
        
        Args:
            force_download: Force re-download even if model exists
            
        Returns:
            Path to downloaded model directory
        """
        model_path = self.models_dir / self.model_name.replace('/', '_')
        
        if model_path.exists() and not force_download:
            logger.info(f"Model already exists at {model_path}")
            return model_path
        
        logger.info(f"Downloading model {self.model_name} from Hugging Face...")
        
        try:
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.models_dir / "cache")
            )
            tokenizer.save_pretrained(str(model_path))
            
            # Download model
            logger.info("Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.models_dir / "cache"),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            model.save_pretrained(str(model_path))
            
            logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        import requests
        base_url = self.config['ollama']['base_url']
        timeout = self.config['ollama'].get('timeout', 5)
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False
    
    def import_to_ollama(self, model_path: Optional[Path] = None) -> bool:
        """
        Import model to Ollama.
        Note: This requires the model to be in GGUF format or using Ollama's import mechanism.
        For now, we'll use Ollama's Modelfile approach or direct import if supported.
        
        Args:
            model_path: Path to model directory (if None, uses downloaded model)
            
        Returns:
            True if successful, False otherwise
        """
        import requests
        import subprocess
        
        base_url = self.config['ollama']['base_url']
        
        if not self.check_ollama_connection():
            logger.error("Ollama is not running. Please start Ollama first.")
            return False
        
        logger.info(f"Attempting to import model {self.ollama_name} to Ollama...")
        
        # Check if model already exists in Ollama
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if any(m['name'] == self.ollama_name for m in models):
                    logger.info(f"Model {self.ollama_name} already exists in Ollama")
                    return True
        except Exception as e:
            logger.warning(f"Could not check existing models: {e}")
        
        # For Hugging Face models, we need to use Ollama's import or create a Modelfile
        # This is a simplified approach - in practice, you may need to convert to GGUF first
        logger.warning(
            "Direct import from Hugging Face format may not be supported by Ollama. "
            "You may need to:\n"
            "1. Convert the model to GGUF format using llama.cpp or similar\n"
            "2. Use Ollama's import command: ollama import <path>\n"
            "3. Or use a model that's already available in Ollama's library"
        )
        
        # Try to use ollama CLI if available
        try:
            # Check if model is available in Ollama's library
            result = subprocess.run(
                ['ollama', 'pull', self.ollama_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                logger.info(f"Successfully pulled model {self.ollama_name} from Ollama library")
                return True
            else:
                logger.warning(f"Model {self.ollama_name} not found in Ollama library")
        except FileNotFoundError:
            logger.warning("Ollama CLI not found in PATH")
        except Exception as e:
            logger.warning(f"Error using Ollama CLI: {e}")
        
        return False
    
    def setup_model(self, force_download: bool = False) -> bool:
        """
        Complete setup: download model and import to Ollama.
        
        Args:
            force_download: Force re-download even if model exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download model
            model_path = self.download_model(force_download=force_download)
            
            # Try to import to Ollama
            # Note: This may require manual steps depending on model format
            imported = self.import_to_ollama(model_path)
            
            if not imported:
                logger.info(
                    "Model downloaded but not yet imported to Ollama. "
                    "You may need to manually convert and import the model."
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            return False

