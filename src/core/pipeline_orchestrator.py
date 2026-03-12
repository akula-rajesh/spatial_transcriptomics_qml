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

# Import packages to register their factories
import src.data_pipeline  # Registers data pipeline factories
import src.models  # Registers model factories
import src.training  # Registers training factories

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

        # Set up experiment
        self.experiment_id = self._create_experiment_id()

        # Initialize result tracker (creates directories automatically)
        results_base_dir = self.config.get('results', {}).get('base_dir', 'results/')
        self.result_tracker = ResultTracker(
            experiment_name=self.experiment_id,
            base_dir=results_base_dir
        )
        self.experiment_dir = self.result_tracker.run_dir

        # Log configuration
        self.result_tracker.log_config(self.config)

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
            # Log all results as metrics
            self._log_final_results(results)

            # Save results to files
            self.result_tracker.save_results()

            # Update status
            self.result_tracker.update_status('completed')
            logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        return results
        
    def _download_data(self) -> None:
        """Download data from configured source(s)."""
        try:
            data_config = self.config.get('data', {})
            pipeline_config = self.config.get('pipeline', {})

            # Create data directories if they don't exist
            input_dir = data_config.get('input_dir', 'data/input/')
            Path(input_dir).mkdir(parents=True, exist_ok=True)
            
            # Support single data_loader or multiple data_loaders
            data_loaders = pipeline_config.get('data_loaders')

            if data_loaders and isinstance(data_loaders, list):
                # Multiple data loaders configured
                logger.info(f"Using {len(data_loaders)} configured data loaders")
                for loader_config in data_loaders:
                    loader_name = loader_config.get('name')
                    enabled = loader_config.get('enabled', True)

                    if not enabled:
                        logger.info(f"Skipping disabled data loader: {loader_name}")
                        continue

                    logger.info(f"Running data loader: {loader_name}")
                    downloader = create_component(
                        ComponentType.DATA_PIPELINE,
                        loader_name,
                        config=self.config
                    )
                    downloader.execute()
            else:
                # Single data loader (default or specified)
                loader_name = pipeline_config.get('data_loader', 'mendeley_downloader')
                logger.info(f"Using data loader: {loader_name}")

                downloader = create_component(
                    ComponentType.DATA_PIPELINE,
                    loader_name,
                    config=self.config  # Pass full config, not just data_config
                )
                downloader.execute()

            logger.info("Data download completed successfully")
            
        except Exception as e:
            logger.error(f"Data download failed: {str(e)}")
            raise
            
    def _process_data(self) -> None:
        """Process raw data into training-ready format using configured steps."""
        try:
            pipeline_config = self.config.get('pipeline', {})

            # Get processing steps from config (with defaults)
            processing_steps = pipeline_config.get('processing_steps')

            if processing_steps and isinstance(processing_steps, list):
                # Use configured processing steps
                logger.info(f"Using {len(processing_steps)} configured processing steps")

                for step_config in processing_steps:
                    step_name = step_config.get('name')
                    enabled = step_config.get('enabled', True)

                    if not enabled:
                        logger.info(f"Skipping disabled processing step: {step_name}")
                        continue

                    logger.info(f"Running processing step: {step_name}")
                    processor = create_component(
                        ComponentType.DATA_PIPELINE,
                        step_name,
                        config=self.config
                    )
                    processor.execute()
            else:
                # Use default processing steps (backward compatibility)
                logger.info("Using default processing steps")

                # File organization
                logger.info("Organizing files...")
                organizer = create_component(
                    ComponentType.DATA_PIPELINE,
                    'file_organizer',
                    config=self.config
                )
                organizer.execute()

                # Stain normalization
                logger.info("Normalizing stains...")
                normalizer = create_component(
                    ComponentType.DATA_PIPELINE,
                    'stain_normalizer',
                    config=self.config
                )
                normalizer.execute()

                # Spatial gene analysis
                logger.info("Analyzing spatial gene patterns...")
                analyzer = create_component(
                    ComponentType.DATA_PIPELINE,
                    'spatial_gene_analyzer',
                    config=self.config
                )
                analyzer.execute()

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
            
            # Save model state
            if self.config.get('models', {}).get('save_best_only', True):
                model_state = {
                    'model_name': active_model,
                    'state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
                    'config': model_config,
                    'training_results': training_results
                }
                self.result_tracker.save_model(model_state, f"{active_model}_final.pth")

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

    def _log_final_results(self, results: Dict[str, Any]) -> None:
        """
        Log final results as metrics in the result tracker.

        Args:
            results: Dictionary containing pipeline results
        """
        # Log training results
        if 'training' in results:
            for model_name, model_results in results['training'].items():
                if isinstance(model_results, dict):
                    # Log best validation loss
                    if 'best_val_loss' in model_results:
                        self.result_tracker.log_metric(
                            f'{model_name}_best_val_loss',
                            model_results['best_val_loss']
                        )
                    # Log training time
                    if 'training_time' in model_results:
                        self.result_tracker.log_metric(
                            f'{model_name}_training_time',
                            model_results['training_time']
                        )

        # Log evaluation results
        if 'evaluation' in results:
            for model_name, eval_results in results['evaluation'].items():
                if isinstance(eval_results, dict):
                    for metric_name, metric_value in eval_results.get('metrics', {}).items():
                        self.result_tracker.log_metric(
                            f'{model_name}_eval_{metric_name}',
                            metric_value
                        )

        logger.info("Final results logged to tracker")

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
