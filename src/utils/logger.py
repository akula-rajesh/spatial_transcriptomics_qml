"""
Logging utilities for the spatial transcriptomics ML pipeline.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'  # Reset color
    
    def format(self, record):
        """Format log record with color."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            
        return super().format(record)

def setup_logging(log_dir: str = 'logs', 
                 log_level: str = 'INFO',
                 console_output: bool = True,
                 file_output: bool = True,
                 max_file_size: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        
    Returns:
        Root logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(colored_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)
        
    # File handler with rotation
    if file_output:
        log_file = log_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)
        
    logger.info("Logging configured successfully")
    logger.info(f"Log file: {log_file if file_output else 'None'}")
    
    return logger

def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger with proper configuration.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Optional override for log level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if log_level:
        logger.setLevel(getattr(logging, log_level.upper()))
        
    return logger

class ExperimentLogger:
    """Logger for experiment results and metrics."""
    
    def __init__(self, experiment_name: str, results_dir: str = 'results/experiments'):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            results_dir: Directory to store experiment results
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir) / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.metrics_file = self.results_dir / "metrics.json"
        self.config_file = self.results_dir / "config.json"
        self.log_file = self.results_dir / "experiment.log"
        
        # Initialize metrics storage
        self.metrics = {}
        self.config = {}
        
        # Setup experiment-specific logger
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        self._setup_experiment_logger()
        
    def _setup_experiment_logger(self):
        """Set up logger for this experiment."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for experiment
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def log_config(self, config: dict):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info("Configuration logged")
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self.metrics:
            self.metrics[name] = []
            
        metric_entry = {'value': value}
        if step is not None:
            metric_entry['step'] = step
            
        self.metrics[name].append(metric_entry)
        
        # Also log to experiment logger
        log_msg = f"Metric {name}: {value:.6f}"
        if step is not None:
            log_msg += f" (step {step})"
        self.logger.info(log_msg)
        
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def save_metrics(self):
        """Save all logged metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info("Metrics saved to file")
        
    def log_info(self, message: str):
        """Log informational message."""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)

# Global experiment logger instance
_current_experiment_logger: Optional[ExperimentLogger] = None

def set_experiment_logger(logger: ExperimentLogger):
    """Set the current experiment logger."""
    global _current_experiment_logger
    _current_experiment_logger = logger

def get_experiment_logger() -> Optional[ExperimentLogger]:
    """Get the current experiment logger."""
    return _current_experiment_logger

def log_experiment_metric(name: str, value: float, step: Optional[int] = None):
    """
    Log metric to current experiment logger.
    
    Args:
        name: Metric name
        value: Metric value
        step: Optional step/epoch number
    """
    if _current_experiment_logger:
        _current_experiment_logger.log_metric(name, value, step)

def log_experiment_metrics(metrics: dict, step: Optional[int] = None):
    """
    Log multiple metrics to current experiment logger.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step/epoch number
    """
    if _current_experiment_logger:
        _current_experiment_logger.log_metrics(metrics, step)

# Convenience functions
def create_experiment_logger(experiment_name: str, 
                           results_dir: str = 'results/experiments') -> ExperimentLogger:
    """
    Create and set experiment logger.
    
    Args:
        experiment_name: Name of the experiment
        results_dir: Directory to store experiment results
        
    Returns:
        Experiment logger instance
    """
    logger = ExperimentLogger(experiment_name, results_dir)
    set_experiment_logger(logger)
    return logger

__all__ = [
    'ColoredFormatter',
    'setup_logging',
    'get_logger',
    'ExperimentLogger',
    'set_experiment_logger',
    'get_experiment_logger',
    'log_experiment_metric',
    'log_experiment_metrics',
    'create_experiment_logger'
]
