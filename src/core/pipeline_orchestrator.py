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
    
    def __init__(self, config_path: Optional[str] = None, resume_path: Optional[str] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Optional path to configuration file (defaults to standard config)
            resume_path: Optional path to a saved .pth checkpoint to resume training from.
                         The checkpoint's state_dict is loaded into the model before
                         the first training epoch.
        """
        self.config_manager = get_config_manager()
        self.factory_registry = get_factory_registry()
        self.result_tracker = None

        # Load main configuration first so we can read resume_path from it
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = load_main_config()

        # Resolve resume_path — CLI arg takes priority over config file value
        config_resume = self.config.get('training', {}).get('resume_path', None)
        if resume_path:
            # CLI --resume was explicitly provided — always use it
            self.resume_path = resume_path
            logger.info(f"Resume path from CLI: {self.resume_path}")
        elif config_resume:
            # Fall back to training.resume_path in pipeline_config.yaml
            self.resume_path = config_resume
            logger.info(f"Resume path from config: {self.resume_path}")
        else:
            self.resume_path = None

        # Set up experiment
        self.experiment_id = self._create_experiment_id()

        # Initialize result tracker (creates directories automatically)
        results_base_dir = self.config.get('results', {}).get('base_dir', 'results/')
        self.result_tracker = ResultTracker(
            experiment_name=self.experiment_id,
            base_dir=results_base_dir
        )
        self.experiment_dir = self.result_tracker.run_dir

        # Register as global tracker so trainers can locate the run directory
        try:
            from src.utils.result_tracker import initialize_tracker
            # Re-use the already-created tracker by pointing the global at it
            import src.utils.result_tracker as _rt_module
            _rt_module._global_tracker = self.result_tracker
        except Exception:
            pass

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

            # Save results to files — pass full results so results.json is complete
            self.result_tracker.save_results(pipeline_results=results)

            # Update status
            self.result_tracker.update_status('completed')
            logger.info("Pipeline execution completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            # Always save whatever results we have, even on failure
            try:
                self._log_final_results(results)
                self.result_tracker.save_results(pipeline_results=results)
                self.result_tracker.update_status('failed')
            except Exception as save_err:
                logger.error(f"Could not save partial results: {save_err}")
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
        """
        Train the configured model.

        FIX: Store the trained trainer instance so _evaluate_model
        can reuse it with best weights intact, instead of creating
        a fresh untrained model.
        """
        results = {}

        try:
            active_model = self.config.get('models', {}).get(
                'active_model', 'classical_efficientnet'
            )
            logger.info(f"Training model: {active_model}")

            model_config = self.config_manager.load_model_config(active_model)

            # Merge pipeline model section
            pipeline_model_section = self.config.get('model', {})
            if pipeline_model_section:
                model_config.update(pipeline_model_section)

            # total_genes is intentionally left as None here.
            # - If resuming: _load_checkpoint() reads aux_head shape from the
            #   checkpoint and calls model.set_aux_head(ckpt_aux_nums) to resize.
            # - If training from scratch: trainer._prepare_data_loaders() calls
            #   model.set_aux_head(dataset.aux_nums) once the dataset is loaded.
            # Setting a hardcoded fallback (e.g. 6250) causes a stale aux_head
            # size mismatch on every resume when the real dataset has a different
            # gene count (e.g. 6216 → aux_nums=5966 ≠ 6000).
            if model_config.get('total_genes') is None:
                model_config['total_genes'] = None   # resolved from dataset/checkpoint

            # Create model
            model = create_component(
                ComponentType.MODEL,
                active_model,
                config=model_config
            )

            # ── Resume from checkpoint if requested ───────────────────
            if self.resume_path:
                self._load_checkpoint(model, self.resume_path)

            # Create trainer
            trainer = create_component(
                ComponentType.TRAINER,
                'supervised_trainer',
                model=model,
                config=self.config
            )

            # Train
            training_results = trainer.train()
            results[active_model] = training_results

            # ── Store trained trainer for evaluation ──────────────────
            # This preserves best_state so evaluate() uses correct weights
            self._trained_trainer = trainer
            self._trained_model   = trainer.model

            # Save model state via result_tracker
            if self.config.get('models', {}).get('save_best_only', True):
                model_state = {
                    'model_name':       active_model,
                    'state_dict':       trainer.model.state_dict(),
                    'config':           model_config,
                    'training_results': training_results,
                    'best_epoch':       trainer.best_epoch,
                }
                self.result_tracker.save_model(
                    model_state, f"{active_model}_final.pth"
                )

            logger.info(f"Model training completed for {active_model}")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

        return results

    def _evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on the test set.

        FIX: Reuse the already-trained trainer (with best_state intact)
        instead of creating a fresh untrained model.
        """
        results = {}

        try:
            active_model = self.config.get('models', {}).get(
                'active_model', 'classical_efficientnet'
            )
            logger.info(f"Evaluating model: {active_model}")

            # ── Prefer the trained trainer from _train_model ──────────
            trained_trainer = getattr(self, '_trained_trainer', None)

            if trained_trainer is not None:
                logger.info(
                    "[PipelineOrchestrator] Reusing trained trainer "
                    f"(best epoch: {trained_trainer.best_epoch + 1}, "
                    f"best val loss: {trained_trainer.best_val_loss:.6f})"
                )
                evaluation_results = trained_trainer.evaluate()
            else:
                # Fall back: create a new trainer and load from checkpoint
                logger.warning(
                    "[PipelineOrchestrator] No trained trainer cached. "
                    "Creating new trainer — will attempt to load saved weights."
                )
                model_config = self.config_manager.load_model_config(active_model)
                pipeline_model_section = self.config.get('model', {})
                if pipeline_model_section:
                    model_config.update(pipeline_model_section)
                if model_config.get('total_genes') is None:
                    model_config['total_genes'] = None   # resolved from checkpoint/dataset

                model   = create_component(ComponentType.MODEL, active_model, config=model_config)
                trainer = create_component(
                    ComponentType.TRAINER, 'supervised_trainer',
                    model=model, config=self.config
                )
                evaluation_results = trainer.evaluate()

            results[active_model] = evaluation_results
            logger.info(f"Model evaluation completed for {active_model}")

        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise

        return results

    def _load_checkpoint(self, model, checkpoint_path: str) -> None:
        """
        Load a saved checkpoint's state_dict into model before training.

        Handles two checkpoint formats:
          1. Plain state_dict  — saved with torch.save(model.state_dict(), path)
          2. Dict with 'state_dict' key — saved by result_tracker.save_model()
             which pickles {'model_name', 'state_dict', 'config', ...}

        Args:
            model: The nn.Module to load weights into (already on device).
            checkpoint_path: Absolute or project-relative path to .pth file.
        """
        import pickle
        import torch

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Determine device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        # Try loading as a pickle file first (result_tracker format)
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)

            if isinstance(payload, dict) and 'state_dict' in payload:
                state_dict = payload['state_dict']
                logger.info(
                    f"Checkpoint format: result_tracker dict "
                    f"(model: {payload.get('model_name', 'unknown')}, "
                    f"best_epoch: {payload.get('best_epoch', '?')})"
                )
            elif isinstance(payload, dict):
                # Assume it IS the state_dict directly
                state_dict = payload
                logger.info("Checkpoint format: plain state_dict (pickle)")
            else:
                raise ValueError(f"Unexpected pickle payload type: {type(payload)}")

        except (pickle.UnpicklingError, Exception):
            # Fall back to torch.load (handles both state_dict and full model)
            raw = torch.load(checkpoint_path, map_location=device)
            if isinstance(raw, dict) and 'state_dict' in raw:
                state_dict = raw['state_dict']
                logger.info("Checkpoint format: torch dict with state_dict key")
            elif isinstance(raw, dict):
                state_dict = raw
                logger.info("Checkpoint format: torch state_dict")
            else:
                # Full model saved with torch.save(model, path)
                logger.info("Checkpoint format: full torch model — copying state_dict")
                state_dict = raw.state_dict()

        # Move state_dict tensors to model's device
        state_dict = {
            k: v.to(device) if hasattr(v, 'to') else v
            for k, v in state_dict.items()
        }

        # ── Fix aux_head size mismatch before loading ──────────────────────
        # The checkpoint was saved with aux_nums computed from the real dataset
        # (e.g. 5966 = 6216 total − 250 main genes × 1.0 aux_ratio).
        # The model was just built using total_genes from config (e.g. 6000),
        # producing a different aux_head shape. PyTorch raises RuntimeError
        # on size mismatch even with strict=False, so we must resize the model
        # aux_head to match the checkpoint BEFORE calling load_state_dict.
        ckpt_aux_weight = state_dict.get("aux_head.weight")  # shape: [aux_nums, feature_dim]
        if ckpt_aux_weight is not None:
            ckpt_aux_nums = ckpt_aux_weight.shape[0]
            model_aux_nums = getattr(model, "aux_nums", None)
            if model_aux_nums != ckpt_aux_nums:
                logger.info(
                    f"  aux_head size mismatch: model has {model_aux_nums}, "
                    f"checkpoint has {ckpt_aux_nums}. "
                    f"Resizing model aux_head → {ckpt_aux_nums} before loading."
                )
                if hasattr(model, "set_aux_head"):
                    model.set_aux_head(ckpt_aux_nums)
                else:
                    # Fallback: directly replace aux_head with correct size
                    import torch.nn as nn
                    feature_dim = ckpt_aux_weight.shape[1]
                    model.aux_head = nn.Linear(feature_dim, ckpt_aux_nums).to(device)
                    model.aux_nums = ckpt_aux_nums
                    logger.info(
                        f"  aux_head replaced directly: "
                        f"Linear({feature_dim} → {ckpt_aux_nums})"
                    )

        # ── Load weights ───────────────────────────────────────────────────
        # strict=False allows partial checkpoint load (missing/extra keys are
        # logged as warnings rather than errors).
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(
                f"  Missing keys in checkpoint ({len(missing)}): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            logger.warning(
                f"  Unexpected keys in checkpoint ({len(unexpected)}): "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )

        logger.info(
            f"✓ Checkpoint loaded successfully — "
            f"resuming training from pretrained weights on {device}"
        )

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

                    # Log best validation metric
                    if 'best_val_metric' in model_results:
                        self.result_tracker.log_metric(
                            f'{model_name}_best_val_metric',
                            model_results['best_val_metric']
                        )

                    # Log training time
                    if 'training_time' in model_results:
                        self.result_tracker.log_metric(
                            f'{model_name}_training_time',
                            model_results['training_time']
                        )

                    # Log epoch-by-epoch training losses
                    if 'train_losses' in model_results:
                        for epoch, loss in enumerate(model_results['train_losses']):
                            self.result_tracker.log_metric(
                                f'{model_name}_train_loss',
                                loss,
                                step=epoch
                            )

                    # Log epoch-by-epoch validation losses
                    if 'val_losses' in model_results:
                        for epoch, loss in enumerate(model_results['val_losses']):
                            self.result_tracker.log_metric(
                                f'{model_name}_val_loss',
                                loss,
                                step=epoch
                            )

                    # Log epoch-by-epoch training metrics (MAE, RMSE, etc.)
                    if 'train_metrics' in model_results:
                        for epoch, metrics_dict in enumerate(model_results['train_metrics']):
                            if isinstance(metrics_dict, dict):
                                for metric_name, metric_value in metrics_dict.items():
                                    if isinstance(metric_value, (int, float)):
                                        self.result_tracker.log_metric(
                                            f'{model_name}_train_{metric_name}',
                                            metric_value,
                                            step=epoch
                                        )

                    # Log epoch-by-epoch validation metrics (MAE, RMSE, correlation)
                    if 'val_metrics' in model_results:
                        for epoch, metrics_dict in enumerate(model_results['val_metrics']):
                            if isinstance(metrics_dict, dict):
                                for metric_name, metric_value in metrics_dict.items():
                                    if isinstance(metric_value, (int, float)):
                                        self.result_tracker.log_metric(
                                            f'{model_name}_val_{metric_name}',
                                            metric_value,
                                            step=epoch
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
