"""
Model downloader for Hugging Face models.
Handles downloading models and preparing them for Ollama.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

logger = logging.getLogger(__name__)

LOG_PATH = "/Users/gaurav/codebase/simplesql/.cursor/debug.log"

def log_debug(session_id, run_id, hypothesis_id, location, message, data=None):
    """Write debug log entry"""
    # #region agent log
    try:
        with open(LOG_PATH, 'a') as f:
            log_entry = {
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "sessionId": session_id,
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data or {}
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion


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
        
        # #region agent log
        session_id = "debug-session"
        run_id = "run1"
        log_debug(session_id, run_id, "A", "model_downloader.py:__init__", "Initializing ModelDownloader", {
            "model_name": self.model_name,
            "has_hf_token_env": "HF_TOKEN" in os.environ,
            "hf_token_length": len(os.environ.get("HF_TOKEN", "")) if "HF_TOKEN" in os.environ else 0,
            "config_has_token": "huggingface" in config and "token" in config.get("huggingface", {}),
            "config_has_model_token": "model" in config and "token" in config.get("model", {})
        })
        # #endregion
    
    def _get_hf_token(self) -> Optional[str]:
        """
        Get Hugging Face token from various sources.
        Priority: config > environment variable > huggingface_hub cache > None
        
        Returns:
            Hugging Face token or None
        """
        # #region agent log
        session_id = "debug-session"
        run_id = "run1"
        # #endregion
        
        # Check config first (huggingface.token or model.token)
        hf_token_config = None
        if isinstance(self.config.get("huggingface"), dict):
            hf_token_config = self.config["huggingface"].get("token")
        if not hf_token_config:
            hf_token_config = self.config.get("model", {}).get("token")
        
        # Check environment variable
        hf_token_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # Check huggingface_hub cache (from `huggingface-cli login`)
        hf_token_cache = None
        try:
            from huggingface_hub import HfFolder
            hf_token_cache = HfFolder.get_token()
        except Exception:
            pass
        
        token = hf_token_config or hf_token_env or hf_token_cache
        
        # #region agent log
        log_debug(session_id, run_id, "A", "model_downloader.py:_get_hf_token", "Token retrieval", {
            "token_from_config": hf_token_config is not None,
            "token_from_env": hf_token_env is not None,
            "token_from_cache": hf_token_cache is not None,
            "final_token_exists": token is not None,
            "token_length": len(token) if token else 0
        })
        # #endregion
        
        return token
        
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
        
        # #region agent log
        session_id = "debug-session"
        run_id = "run1"
        token_to_use = self._get_hf_token()
        log_debug(session_id, run_id, "B", "model_downloader.py:download_model", "Before tokenizer download", {
            "token_to_use_exists": token_to_use is not None,
            "token_length": len(token_to_use) if token_to_use else 0
        })
        # #endregion
        
        try:
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            # #region agent log
            log_debug(session_id, run_id, "C", "model_downloader.py:download_model", "Calling AutoTokenizer.from_pretrained", {
                "model_name": self.model_name,
                "token_provided": token_to_use is not None
            })
            # #endregion
            
            tokenizer_kwargs = {
                "cache_dir": str(self.models_dir / "cache")
            }
            if token_to_use:
                tokenizer_kwargs["token"] = token_to_use
                # #region agent log
                log_debug(session_id, run_id, "C", "model_downloader.py:download_model", "Adding token to tokenizer kwargs")
                # #endregion
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )
            tokenizer.save_pretrained(str(model_path))
            
            # Download model
            logger.info("Downloading model...")
            # #region agent log
            log_debug(session_id, run_id, "D", "model_downloader.py:download_model", "Calling AutoModelForCausalLM.from_pretrained", {
                "model_name": self.model_name,
                "token_provided": token_to_use is not None
            })
            # #endregion
            
            model_kwargs = {
                "cache_dir": str(self.models_dir / "cache"),
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True
            }
            if token_to_use:
                model_kwargs["token"] = token_to_use
                # #region agent log
                log_debug(session_id, run_id, "D", "model_downloader.py:download_model", "Adding token to model kwargs")
                # #endregion
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            model.save_pretrained(str(model_path))
            
            logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
            
        except Exception as e:
            # #region agent log
            log_debug(session_id, run_id, "E", "model_downloader.py:download_model", "Exception during download", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "token_was_provided": token_to_use is not None
            })
            # #endregion
            
            # Check if this is an authentication error
            error_str = str(e).lower()
            if ("gated repo" in error_str or "401" in error_str or "authenticated" in error_str or 
                "access" in error_str and "restricted" in error_str) and not token_to_use:
                logger.error(f"Error downloading model: {e}")
                logger.error("\n" + "="*70)
                logger.error("AUTHENTICATION REQUIRED")
                logger.error("="*70)
                logger.error("This model requires Hugging Face authentication.")
                logger.error("\nTo fix this, choose ONE of the following methods:")
                logger.error("\n1. Set HF_TOKEN environment variable:")
                logger.error("   export HF_TOKEN=your_token_here")
                logger.error("\n2. Add token to config file (config.yaml or your config):")
                logger.error("   huggingface:")
                logger.error("     token: your_token_here")
                logger.error("\n3. Login via huggingface-cli:")
                logger.error("   pip install huggingface_hub")
                logger.error("   huggingface-cli login")
                logger.error("\nTo get your token:")
                logger.error("1. Go to https://huggingface.co/settings/tokens")
                logger.error("2. Create a new token (read access is sufficient)")
                logger.error("3. Accept the model's terms at https://huggingface.co/google/gemma-2b-it")
                logger.error("="*70)
            else:
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

