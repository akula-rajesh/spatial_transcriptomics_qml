"""
Main pipeline orchestrator that controls the execution flow of the ML pipeline.
"""

import os
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.utils.config_manager import get_config_manager, load_main_config
from src.core.factory_registry import get_factory_registry, ComponentType, create_component
from src.utils.result_tracker import ResultTracker

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the execution of the ML pipeline based on configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Optional path to configuration file (defaults to standard config)
        """
        self.config_manager = get_config_manager()
        self.factory_registry = get_factory_registry()
        self.result_tracker = None
        
        # Load main configuration
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = load_main_config()
            
        # Initialize result tracker
        results_base_dir = self.config.get('results', {}).get('base_dir', 'results/')
        self.result_tracker = ResultTracker(results_base_dir)
        
        # Set up experiment
        self.experiment_id = self._create_experiment_id()
        self.experiment_dir = self.result_tracker.create_experiment_directory(self.experiment_id)
        
        logger.info(f"Initialized pipeline orchestrator for experiment: {self.experiment_id}")
        
    def _create_experiment_id(self) -> str:
        """
        Create a unique experiment ID based on configuration.
        
        Returns:
            Unique experiment identifier
        """
        naming_strategy = self.config.get('results', {}).get('experiment_naming', 'timestamp')
        
        if naming_strategy == 'timestamp':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"experiment_{timestamp}"
        else:
            # Custom naming strategy could be implemented here
            return f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete ML pipeline according to configuration.
        
        Returns:
            Dictionary containing pipeline results and metrics
        """
        logger.info("Starting pipeline execution")
        results = {}
        
        try:
            # Step 1: Download data if configured
            if self.config.get('pipeline', {}).get('download_data', False):
                logger.info("Step 1: Downloading data")
                self._download_data()
                
            # Step 2: Process data if configured
            if self.config.get('pipeline', {}).get('process_data', False):
                logger.info("Step 2: Processing data")
                self._process_data()
                
            # Step 3: Train model if configured
            if self.config.get('pipeline', {}).get('train_model', False):
                logger.info("Step 3: Training model")
                training_results = self._train_model()
                results['training'] = training_results
                
            # Step 4: Evaluate model if configured
            if self.config.get('pipeline', {}).get('evaluate_model', False):
                logger.info("Step 4: Evaluating model")
                evaluation_results = self._evaluate_model()
                results['evaluation'] = evaluation_results
                
            # Step 5: Compare results if configured
            if self.config.get('pipeline', {}).get('compare_results', False):
                logger.info("Step 5: Comparing results")
                comparison_results = self._compare_results()
                results['comparison'] = comparison_results
                
            # Save final results
            self.result_tracker.save_experiment_results(self.experiment_id, results)
            logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        return results
        
    def _download_data(self) -> None:
        """Download data from configured source."""
        try:
            # Create data directories if they don't exist
            data_config = self.config.get('data', {})
            input_dir = data_config.get('input_dir', 'data/input/')
            Path(input_dir).mkdir(parents=True, exist_ok=True)
            
            # Get downloader from factory
            downloader = create_component(
                ComponentType.DATA_PIPELINE,
                'mendeley_downloader',
                config=data_config
            )
            
            # Download data
            downloader.download()
            
            logger.info("Data download completed successfully")
            
        except Exception as e:
            logger.error(f"Data download failed: {str(e)}")
            raise
            
    def _process_data(self) -> None:
        """Process raw data into training-ready format."""
        try:
            data_config = self.config.get('data', {})
            
            # File organization
            organizer = create_component(
                ComponentType.DATA_PIPELINE,
                'file_organizer',
                config=data_config
            )
            organizer.organize_files()
            
            # Stain normalization
            normalizer = create_component(
                ComponentType.DATA_PIPELINE,
                'stain_normalizer',
                config=data_config
            )
            normalizer.normalize_stains()
            
            # Spatial gene analysis
            analyzer = create_component(
                ComponentType.DATA_PIPELINE,
                'spatial_gene_analyzer',
                config=self.config
            )
            analyzer.analyze_spatial_patterns()
            
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise
            
    def _train_model(self) -> Dict[str, Any]:
        """Train the configured model(s)."""
        results = {}
        
        try:
            # Get active model from configuration
            active_model = self.config.get('models', {}).get('active_model', 'classical_efficientnet')
            
            logger.info(f"Training model: {active_model}")
            
            # Load model configuration
            model_config = self.config_manager.load_model_config(active_model)
            
            # Create model using factory
            model = create_component(
                ComponentType.MODEL,
                active_model,
                config=model_config
            )
            
            # Create trainer using factory
            trainer = create_component(
                ComponentType.TRAINER,
                'supervised_trainer',
                model=model,
                config=self.config
            )
            
            # Train the model
            training_results = trainer.train()
            results[active_model] = training_results
            
            # Save model checkpoint
            if self.config.get('models', {}).get('save_best_only', True):
                self.result_tracker.save_model_checkpoint(
                    self.experiment_id, 
                    active_model, 
                    model
                )
                
            logger.info(f"Model training completed for {active_model}")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
            
        return results
        
    def _evaluate_model(self) -> Dict[str, Any]:
        """Evaluate trained model(s) on test data."""
        results = {}
        
        try:
            # Get active model from configuration
            active_model = self.config.get('models', {}).get('active_model', 'classical_efficientnet')
            
            logger.info(f"Evaluating model: {active_model}")
            
            # Load model configuration
            model_config = self.config_manager.load_model_config(active_model)
            
            # Create model using factory
            model = create_component(
                ComponentType.MODEL,
                active_model,
                config=model_config
            )
            
            # Create trainer for evaluation
            trainer = create_component(
                ComponentType.TRAINER,
                'supervised_trainer',
                model=model,
                config=self.config
            )
            
            # Evaluate the model
            evaluation_results = trainer.evaluate()
            results[active_model] = evaluation_results
            
            logger.info(f"Model evaluation completed for {active_model}")
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
            
        return results
        
    def _compare_results(self) -> Dict[str, Any]:
        """Compare results between different models."""
        results = {}
        
        try:
            models_to_compare = self.config.get('models', {}).get('compare_models', [])
            
            if len(models_to_compare) < 2:
                logger.info("Not enough models to compare")
                return results
                
            logger.info(f"Comparing models: {models_to_compare}")
            
            comparison_metrics = {}
            
            # Collect metrics for each model
            for model_name in models_to_compare:
                try:
                    # This would typically load saved results from previous runs
                    # For now, we'll just simulate comparison data
                    comparison_metrics[model_name] = {
                        'aMAE': 0.0,
                        'aRMSE': 0.0,
                        'aCC': 0.0,
                        'training_time': 0.0,
                        'inference_time': 0.0
                    }
                except Exception as e:
                    logger.warning(f"Could not collect metrics for {model_name}: {str(e)}")
                    
            results['metrics'] = comparison_metrics
            
            # Determine best performing model based on primary metric
            primary_metric = self.config.get('training', {}).get('validation_metric', 'loss')
            best_model = self._determine_best_model(comparison_metrics, primary_metric)
            results['best_model'] = best_model
            
            logger.info(f"Model comparison completed. Best model: {best_model}")
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            raise
            
        return results
        
    def _determine_best_model(self, metrics: Dict[str, Dict[str, float]], primary_metric: str) -> str:
        """
        Determine the best performing model based on primary metric.
        
        Args:
            metrics: Dictionary of metrics for each model
            primary_metric: Primary metric to use for comparison
            
        Returns:
            Name of the best performing model
        """
        if not metrics:
            return "unknown"
            
        # Convert metric name to standardized format if needed
        metric_key = primary_metric.lower()
        if metric_key in ['loss', 'mae', 'rmse']:
            # Lower is better
            best_model = min(metrics.items(), key=lambda x: x[1].get(metric_key, float('inf')))[0]
        else:
            # Higher is better (assumed for correlation coefficients)
            best_model = max(metrics.items(), key=lambda x: x[1].get(metric_key, float('-inf')))[0]
            
        return best_model
        
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about the current experiment.
        
        Returns:
            Dictionary containing experiment information
        """
        return {
            'experiment_id': self.experiment_id,
            'experiment_dir': str(self.experiment_dir),
            'config': self.config,
            'start_time': datetime.now().isoformat()
        }

def run_pipeline_from_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline with a given configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary containing pipeline results
    """
    orchestrator = PipelineOrchestrator(config_path)
    return orchestrator.run_pipeline()

def main():
    """Main entry point for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spatial Transcriptomics ML Pipeline")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run the pipeline
        orchestrator = PipelineOrchestrator(args.config)
        results = orchestrator.run_pipeline()
        
        logger.info("Pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
