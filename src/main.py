#!/usr/bin/env python3
"""
Main entry point for the spatial transcriptomics ML pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import yaml
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.factory import ModelFactory
from src.training.factory import TrainerFactory
from src.training.cross_validator import run_cross_validation
from src.utils.result_tracker import initialize_tracker, get_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/main.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[Any, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
    logger.info(f"Loaded configuration from {config_path}")
    return config

def setup_environment(config: Dict[Any, Any]) -> None:
    """
    Setup environment based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Create necessary directories
    dirs_to_create = [
        'logs',
        config.get('results', {}).get('base_dir', 'results'),
        'data'
    ]
    
    for dir_path in dirs_to_create:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(config.get('random_seed', 42))
    
    logger.info("Environment setup completed")

def run_single_experiment(config: Dict[Any, Any]) -> None:
    """
    Run a single training experiment.
    
    Args:
        config: Configuration dictionary
    """
    experiment_name = config.get('experiment_name', 'spatial_transcriptomics_experiment')
    
    # Initialize result tracker
    tracker = initialize_tracker(experiment_name)
    tracker.log_config(config)
    tracker.update_status('running')
    
    try:
        # Create model
        model_name = config.get('model', {}).get('name', 'classical_efficientnet')
        logger.info(f"Creating model: {model_name}")
        model = ModelFactory.create_model(model_name, config.get('model', {}))
        
        # Create trainer
        trainer_name = config.get('trainer', {}).get('name', 'supervised_trainer')
        logger.info(f"Creating trainer: {trainer_name}")
        trainer = TrainerFactory.create_trainer(trainer_name, model, config.get('trainer', {}))
        
        # Train model
        logger.info("Starting model training")
        training_results = trainer.train()
        
        # Evaluate model
        logger.info("Starting model evaluation")
        evaluation_results = trainer.evaluate()
        
        # Log results
        tracker.log_metrics({
            'final_train_loss': training_results.get('best_val_loss', 0),
            'final_val_loss': evaluation_results.get('loss', 0),
            'final_val_mae': evaluation_results.get('metrics', {}).get('mae', 0),
            'training_time': training_results.get('training_time', 0)
        })
        
        # Save final model
        final_model_path = f"results/{experiment_name}/final_model.pth"
        # In a real implementation, you would save the actual model state
        # torch.save(model.state_dict(), final_model_path)
        
        # Save results summary
        results_summary = {
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
        results_file = Path(f"results/{experiment_name}/results_summary.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        tracker.update_status('completed')
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        tracker.update_status('failed')
        raise

def run_cross_validation_experiment(config: Dict[Any, Any]) -> None:
    """
    Run cross-validation experiment.
    
    Args:
        config: Configuration dictionary
    """
    experiment_name = config.get('experiment_name', 'spatial_transcriptomics_cv')
    
    # Initialize result tracker
    tracker = initialize_tracker(experiment_name)
    tracker.log_config(config)
    tracker.update_status('running')
    
    try:
        logger.info("Starting cross-validation experiment")
        
        # Run cross-validation
        cv_results = run_cross_validation(config)
        
        # Log results
        for metric in ['loss', 'mae', 'rmse', 'correlation_coefficient']:
            mean_val = cv_results.get(f'{metric}_mean', 0)
            std_val = cv_results.get(f'{metric}_std', 0)
            tracker.log_metrics({
                f'cv_{metric}_mean': mean_val,
                f'cv_{metric}_std': std_val
            })
            
        tracker.update_status('completed')
        logger.info("Cross-validation experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Cross-validation experiment failed: {str(e)}", exc_info=True)
        tracker.update_status('failed')
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Spatial Transcriptomics ML Pipeline")
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'cross_validate'],
                       default='train', help='Execution mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup environment
        setup_environment(config)
        
        # Run experiment based on mode
        if args.mode == 'train':
            run_single_experiment(config)
        elif args.mode == 'cross_validate':
            run_cross_validation_experiment(config)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()
