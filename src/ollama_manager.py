"""
Ollama API manager for model inference.
Handles communication with Ollama and text-to-SQL prompt formatting.
"""

import logging
import requests
import time
from typing import Optional, Dict, Any, List
from .utils import extract_sql_from_response

logger = logging.getLogger(__name__)


class OllamaManager:
    """Manage Ollama API interactions for text-to-SQL inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_url = config['ollama']['base_url']
        self.model_name = config['model']['ollama_name']
        self.timeout = config['ollama'].get('timeout', 300)
        self.max_tokens = config['model'].get('max_tokens', 512)
        self.temperature = config['model'].get('temperature', 0.1)
        self.top_p = config['model'].get('top_p', 0.9)
        
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def check_model_exists(self) -> bool:
        """
        Check if the configured model exists in Ollama.
        
        Returns:
            True if model exists, False otherwise
        """
        models = self.list_models()
        return self.model_name in models
    
    def format_text_to_sql_prompt(
        self,
        question: str,
        schema: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format prompt for text-to-SQL task.
        
        Args:
            question: Natural language question
            schema: Database schema information
            examples: Optional few-shot examples
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append(
            "You are a SQL expert. Generate SQL queries based on natural language questions. "
            "Return only the SQL query without any explanation."
        )
        
        # Add schema if provided
        if schema:
            prompt_parts.append(f"\nDatabase Schema:\n{schema}")
        
        # Add examples if provided
        if examples:
            prompt_parts.append("\nExamples:")
            for ex in examples:
                prompt_parts.append(f"Question: {ex['question']}")
                prompt_parts.append(f"SQL: {ex['sql']}\n")
        
        # Add the actual question
        prompt_parts.append(f"\nQuestion: {question}")
        prompt_parts.append("SQL:")
        
        return "\n".join(prompt_parts)
    
    def generate_sql(
        self,
        question: str,
        schema: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            schema: Database schema information
            examples: Optional few-shot examples
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated SQL query or None if failed
        """
        if not self.check_connection():
            logger.error("Ollama is not running or not accessible")
            return None
        
        if not self.check_model_exists():
            logger.error(f"Model {self.model_name} not found in Ollama")
            return None
        
        prompt = self.format_text_to_sql_prompt(question, schema, examples)
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "num_predict": self.max_tokens,
                        }
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    
                    # Extract SQL from response
                    sql = extract_sql_from_response(response_text)
                    logger.debug(f"Generated SQL: {sql}")
                    return sql
                else:
                    logger.warning(f"Ollama API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error generating SQL (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error("Failed to generate SQL after all retries")
        return None
    
    def generate_batch(
        self,
        questions: List[str],
        schemas: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> List[Optional[str]]:
        """
        Generate SQL queries for multiple questions.
        
        Args:
            questions: List of natural language questions
            schemas: Optional list of schemas (one per question)
            examples: Optional few-shot examples
            
        Returns:
            List of generated SQL queries (None for failed generations)
        """
        results = []
        schemas = schemas or [None] * len(questions)
        
        for i, (question, schema) in enumerate(zip(questions, schemas)):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            sql = self.generate_sql(question, schema, examples)
            results.append(sql)
        
        return results

