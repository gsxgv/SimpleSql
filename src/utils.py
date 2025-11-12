"""
Utility functions for configuration loading and common helpers.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def ensure_dir(path: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_sql_query(sql: str) -> str:
    """
    Format SQL query for better readability.
    
    Args:
        sql: Raw SQL query string
        
    Returns:
        Formatted SQL query
    """
    import sqlparse
    try:
        formatted = sqlparse.format(sql, reindent=True, keyword_case='upper')
        return formatted.strip()
    except Exception:
        return sql.strip()


def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from model response.
    Handles cases where response may contain markdown code blocks or extra text.
    
    Args:
        response: Model response text
        
    Returns:
        Extracted SQL query
    """
    response = response.strip()
    
    # Remove markdown code blocks if present
    if '```sql' in response.lower():
        start = response.lower().find('```sql') + 6
        end = response.lower().find('```', start)
        if end != -1:
            response = response[start:end].strip()
    elif '```' in response:
        start = response.find('```') + 3
        end = response.find('```', start)
        if end != -1:
            response = response[start:end].strip()
    
    # Find SQL keywords to locate the actual query
    sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']
    lines = response.split('\n')
    sql_start = 0
    for i, line in enumerate(lines):
        if any(line.strip().upper().startswith(kw) for kw in sql_keywords):
            sql_start = i
            break
    
    sql = '\n'.join(lines[sql_start:]).strip()
    
    # Remove trailing semicolons if they're not part of the query
    if sql.endswith(';') and sql.count(';') == 1:
        sql = sql.rstrip(';')
    
    return sql.strip()


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent

