"""
Benchmark evaluator for text-to-SQL models.
Handles loading benchmarks, executing queries, and calculating metrics.
"""

import os
import json
import logging
import sqlparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from tqdm import tqdm

from .ollama_manager import OllamaManager
from .utils import format_sql_query

logger = logging.getLogger(__name__)


class SQLEvaluator:
    """Evaluate text-to-SQL models on benchmarks."""
    
    def __init__(self, config: Dict[str, Any], ollama_manager: OllamaManager):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
            ollama_manager: OllamaManager instance
        """
        self.config = config
        self.ollama_manager = ollama_manager
        self.benchmark_name = config['benchmark']['name']
        self.database_path = Path(config['benchmark']['database_path'])
        self.data_path = Path(config['benchmark']['data_path'])
        self.timeout = config['evaluation'].get('timeout', 30)
        self.metrics = config['evaluation'].get('metrics', ['exact_match', 'execution_accuracy'])
        self.results_dir = Path(config['paths']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_spider_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load Spider benchmark dataset.
        
        Args:
            split: Dataset split (train, dev, test)
            
        Returns:
            List of examples with question, sql, db_id, etc.
        """
        data_file = self.data_path / f"{split}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Spider dataset file not found: {data_file}. "
                "Please download the Spider dataset first."
            )
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} examples from {split} split")
        return data
    
    def load_database_schema(self, db_id: str) -> str:
        """
        Load database schema information.
        
        Args:
            db_id: Database identifier
            
        Returns:
            Schema description string
        """
        schema_file = self.data_path / "tables.json"
        
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                tables = json.load(f)
            
            db_info = next((t for t in tables if t['db_id'] == db_id), None)
            if db_info:
                schema_parts = [f"Database: {db_id}"]
                for table in db_info.get('table_names_original', []):
                    schema_parts.append(f"\nTable: {table}")
                    # Add columns if available
                    if 'column_names_original' in db_info:
                        cols = [
                            col[1] for col in db_info['column_names_original']
                            if col[0] == db_info['table_names_original'].index(table)
                        ]
                        schema_parts.append(f"  Columns: {', '.join(cols)}")
                
                return "\n".join(schema_parts)
        
        return f"Database: {db_id}"
    
    def create_database_connection(self, db_id: str):
        """
        Create database connection for a specific database.
        
        Args:
            db_id: Database identifier
            
        Returns:
            SQLAlchemy engine
        """
        db_path = self.database_path / db_id / f"{db_id}.sqlite"
        
        if db_path.exists():
            # SQLite database
            return create_engine(f"sqlite:///{db_path}")
        else:
            # Try MySQL connection (for Spider)
            mysql_config = {
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'user': os.getenv('MYSQL_USER', 'root'),
                'password': os.getenv('MYSQL_PASSWORD', ''),
                'database': db_id
            }
            connection_string = (
                f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}"
                f"@{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
            )
            return create_engine(connection_string)
    
    def normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for comparison.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL query
        """
        try:
            # Parse and format SQL
            parsed = sqlparse.parse(sql)[0]
            normalized = sqlparse.format(
                str(parsed),
                reindent=True,
                keyword_case='upper',
                strip_comments=True
            )
            # Remove extra whitespace
            normalized = ' '.join(normalized.split())
            return normalized.strip()
        except Exception as e:
            logger.debug(f"Error normalizing SQL: {e}")
            return sql.strip().upper()
    
    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted SQL exactly matches ground truth.
        
        Args:
            predicted: Predicted SQL query
            ground_truth: Ground truth SQL query
            
        Returns:
            True if exact match, False otherwise
        """
        pred_norm = self.normalize_sql(predicted)
        gt_norm = self.normalize_sql(ground_truth)
        return pred_norm == gt_norm
    
    def execution_match(
        self,
        predicted: str,
        ground_truth: str,
        db_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if predicted SQL execution matches ground truth execution.
        
        Args:
            predicted: Predicted SQL query
            ground_truth: Ground truth SQL query
            db_id: Database identifier
            
        Returns:
            Tuple of (match, error_message)
        """
        try:
            engine = self.create_database_connection(db_id)
            
            with engine.connect() as conn:
                # Execute ground truth query
                try:
                    gt_result = pd.read_sql(text(ground_truth), conn)
                except Exception as e:
                    logger.debug(f"Error executing ground truth: {e}")
                    return False, str(e)
                
                # Execute predicted query
                try:
                    pred_result = pd.read_sql(text(predicted), conn)
                except Exception as e:
                    logger.debug(f"Error executing predicted query: {e}")
                    return False, str(e)
                
                # Compare results
                # Sort by all columns for comparison
                try:
                    gt_sorted = gt_result.sort_values(by=list(gt_result.columns)).reset_index(drop=True)
                    pred_sorted = pred_result.sort_values(by=list(pred_result.columns)).reset_index(drop=True)
                    
                    # Compare DataFrames
                    match = gt_sorted.equals(pred_sorted)
                    return match, None
                except Exception as e:
                    logger.debug(f"Error comparing results: {e}")
                    return False, str(e)
                    
        except Exception as e:
            logger.debug(f"Error in execution match: {e}")
            return False, str(e)
    
    def evaluate_example(
        self,
        example: Dict[str, Any],
        max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single example.
        
        Args:
            example: Example dictionary with question, sql, db_id, etc.
            max_examples: Maximum number of examples (for progress tracking)
            
        Returns:
            Evaluation result dictionary
        """
        question = example['question']
        ground_truth_sql = example['sql']
        db_id = example['db_id']
        
        # Load schema
        schema = self.load_database_schema(db_id)
        
        # Generate SQL
        predicted_sql = self.ollama_manager.generate_sql(question, schema=schema)
        
        if predicted_sql is None:
            return {
                'question': question,
                'ground_truth': ground_truth_sql,
                'predicted': None,
                'exact_match': False,
                'execution_match': False,
                'error': 'Failed to generate SQL'
            }
        
        # Calculate metrics
        result = {
            'question': question,
            'ground_truth': ground_truth_sql,
            'predicted': predicted_sql,
            'db_id': db_id
        }
        
        if 'exact_match' in self.metrics:
            result['exact_match'] = self.exact_match(predicted_sql, ground_truth_sql)
        
        if 'execution_accuracy' in self.metrics:
            exec_match, error = self.execution_match(predicted_sql, ground_truth_sql, db_id)
            result['execution_match'] = exec_match
            if error:
                result['execution_error'] = error
        
        return result
    
    def evaluate(
        self,
        split: str = "test",
        max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on benchmark dataset.
        
        Args:
            split: Dataset split to evaluate on
            max_examples: Maximum number of examples to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary
        """
        # Load dataset
        if self.benchmark_name.lower() == 'spider':
            examples = self.load_spider_dataset(split)
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")
        
        if max_examples:
            examples = examples[:max_examples]
        
        logger.info(f"Evaluating on {len(examples)} examples...")
        
        results = []
        for i, example in enumerate(tqdm(examples, desc="Evaluating")):
            try:
                result = self.evaluate_example(example)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {e}")
                results.append({
                    'question': example.get('question', ''),
                    'ground_truth': example.get('sql', ''),
                    'predicted': None,
                    'error': str(e)
                })
        
        # Calculate aggregate metrics
        metrics = {}
        if 'exact_match' in self.metrics:
            exact_matches = sum(1 for r in results if r.get('exact_match', False))
            metrics['exact_match_accuracy'] = exact_matches / len(results) if results else 0
        
        if 'execution_accuracy' in self.metrics:
            exec_matches = sum(1 for r in results if r.get('execution_match', False))
            metrics['execution_accuracy'] = exec_matches / len(results) if results else 0
        
        # Save results
        output_file = self.results_dir / f"evaluation_{split}_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': results,
                'config': {
                    'benchmark': self.benchmark_name,
                    'split': split,
                    'num_examples': len(results)
                }
            }, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {output_file}")
        logger.info(f"Metrics: {metrics}")
        
        return {
            'metrics': metrics,
            'results': results,
            'output_file': str(output_file)
        }

