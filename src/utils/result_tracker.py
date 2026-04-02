"""
Result tracking utilities for experiment management.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)

class ResultTracker:
    """Tracks and manages experiment results."""
    
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        """
        Initialize result tracker.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for storing results
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.experiment_dir / self.timestamp
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking structures
        self.metrics = defaultdict(list)
        self.config = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'status': 'initialized'
        }
        
        # Log files
        self.log_file = self.run_dir / "experiment.log"
        self.metrics_file = self.run_dir / "metrics.json"
        self.config_file = self.run_dir / "config.json"
        self.metadata_file = self.run_dir / "metadata.json"
        
        logger.info(f"Initialized ResultTracker for experiment '{experiment_name}'")
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info("Configuration logged successfully")
        except Exception as e:
            logger.error(f"Failed to log configuration: {str(e)}")
            
    def log_metric(self, name: str, value: Union[float, int], step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Name of the metric
            value: Metric value
            step: Training step (optional)
        """
        entry = {
            'value': float(value),
            'timestamp': datetime.now().isoformat()
        }
        if step is not None:
            entry['step'] = step
            
        self.metrics[name].append(entry)
        
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def log_metadata(self, key: str, value: Any) -> None:
        """
        Log metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        
    def update_status(self, status: str) -> None:
        """
        Update experiment status.
        
        Args:
            status: New status
        """
        self.metadata['status'] = status
        logger.info(f"Experiment status updated to: {status}")
        
    def save_checkpoint(self, data: Any, filename: str) -> None:
        """
        Save checkpoint data.
        
        Args:
            data: Data to save
            filename: Filename for checkpoint
        """
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        filepath = checkpoint_dir / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            
    def save_model(self, model_state: Dict[str, Any], filename: str = "model.pth") -> None:
        """
        Save model state.
        
        Args:
            model_state: Model state dictionary
            filename: Model filename
        """
        models_dir = self.run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        filepath = models_dir / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            
    def save_results(self, pipeline_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Save all tracked results.

        Args:
            pipeline_results: Optional full pipeline results dict to include in results.json
        """
        try:
            # 1. Save metrics.json — raw per-step metric log
            with open(self.metrics_file, 'w') as f:
                json.dump(dict(self.metrics), f, indent=2, default=str)

            # 2. Save metadata.json
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)

            # 3. Build and save results.json — comprehensive summary for visualisation
            results_json = self._build_results_json(pipeline_results)
            results_file = self.run_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results_json, f, indent=2, default=str)

            logger.info(f"Results saved to {self.run_dir}")
            logger.info(f"  metrics.json  → {self.metrics_file}")
            logger.info(f"  results.json  → {results_file}")
            logger.info(f"  metadata.json → {self.metadata_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

    def _build_results_json(self, pipeline_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a comprehensive results.json dict suitable for visualisation.

        Structure:
          {
            experiment_name, timestamp, status,
            training: { <model_name>: { epochs, losses, metrics, best_epoch, ... } },
            evaluation: { <model_name>: { loss, amae, armse, acc, ... } },
            metrics_summary: { <metric_name>: { last, min, max, mean } }
          }
        """
        results = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'status': self.metadata.get('status', 'unknown'),
            'run_dir': str(self.run_dir),
        }

        # ── Training history ──────────────────────────────────────────
        training_section: Dict[str, Any] = {}

        # Pull from pipeline_results if provided
        if pipeline_results and 'training' in pipeline_results:
            for model_name, model_res in pipeline_results['training'].items():
                if not isinstance(model_res, dict):
                    continue
                training_section[model_name] = {
                    'best_epoch':      model_res.get('best_epoch', model_res.get('final_epoch', 0)),
                    'best_val_loss':   model_res.get('best_val_loss', None),
                    'best_val_metric': model_res.get('best_val_metric', None),
                    'training_time_s': model_res.get('training_time', None),
                    'total_epochs':    len(model_res.get('train_losses', [])),
                    # Per-epoch arrays (safe for JSON — plain lists of floats)
                    'train_losses':    [float(v) for v in model_res.get('train_losses', [])],
                    'val_losses':      [float(v) for v in model_res.get('val_losses', [])],
                    # Per-epoch metric dicts
                    'train_metrics':   self._sanitise_metric_history(model_res.get('train_metrics', [])),
                    'val_metrics':     self._sanitise_metric_history(model_res.get('val_metrics', [])),
                }

        # Also rebuild from the metrics log (works even when pipeline_results is None)
        if not training_section:
            training_section = self._rebuild_training_from_metrics()

        if training_section:
            results['training'] = training_section

        # ── Evaluation results ────────────────────────────────────────
        if pipeline_results and 'evaluation' in pipeline_results:
            eval_section: Dict[str, Any] = {}
            for model_name, eval_res in pipeline_results['evaluation'].items():
                if isinstance(eval_res, dict):
                    eval_section[model_name] = {
                        'loss':                    eval_res.get('loss', None),
                        'metrics':                 eval_res.get('metrics', {}),
                    }
            if eval_section:
                results['evaluation'] = eval_section

        # ── Flat metrics summary (last / min / max / mean per metric) ─
        summary: Dict[str, Any] = {}
        for name, entries in self.metrics.items():
            numeric = [e['value'] for e in entries
                       if isinstance(e.get('value'), (int, float)) and not np.isnan(e['value'])]
            if numeric:
                summary[name] = {
                    'last':  numeric[-1],
                    'min':   float(np.min(numeric)),
                    'max':   float(np.max(numeric)),
                    'mean':  float(np.mean(numeric)),
                    'count': len(numeric),
                }
        if summary:
            results['metrics_summary'] = summary

        return results

    @staticmethod
    def _sanitise_metric_history(history: list) -> list:
        """Convert list of metric dicts to JSON-safe list of plain dicts."""
        out = []
        for item in history:
            if isinstance(item, dict):
                out.append({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                             for k, v in item.items()})
        return out

    def _rebuild_training_from_metrics(self) -> Dict[str, Any]:
        """
        Rebuild per-model training history from self.metrics when
        pipeline_results is not available (e.g. called after failure).
        """
        training: Dict[str, Any] = {}
        for key, entries in self.metrics.items():
            # Keys look like  "classical_efficientnet_train_loss"
            parts = key.split('_')
            if len(parts) < 3:
                continue
            # Find the split between model name and metric name
            for split in range(1, len(parts)):
                suffix = '_'.join(parts[split:])
                if suffix in ('train_loss', 'val_loss', 'train_amae', 'val_amae',
                              'val_armse', 'train_armse', 'val_correlation_coefficient',
                              'best_val_loss', 'training_time'):
                    model_name = '_'.join(parts[:split])
                    if model_name not in training:
                        training[model_name] = {}
                    step_values = sorted(entries, key=lambda e: e.get('step', 0))
                    training[model_name][suffix] = [float(e['value']) for e in step_values]
                    break
        return training

    def save_plots(self, plot_types: Optional[List[str]] = None) -> None:
        """
        Save plots of tracked metrics.
        
        Args:
            plot_types: List of plot types to generate (default: all)
        """
        if plot_types is None:
            plot_types = ['line', 'heatmap']
            
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Get numeric metrics
        numeric_metrics = {}
        for name, values in self.metrics.items():
            numeric_values = []
            steps = []
            for entry in values:
                if isinstance(entry['value'], (int, float)) and not np.isnan(entry['value']):
                    numeric_values.append(entry['value'])
                    steps.append(entry.get('step', len(steps)))
            if numeric_values:
                numeric_metrics[name] = {'values': numeric_values, 'steps': steps}
                
        if not numeric_metrics:
            logger.warning("No numeric metrics to plot")
            return
            
        try:
            # Line plots
            if 'line' in plot_types:
                self._save_line_plots(numeric_metrics, plots_dir)
                
            # Heatmap
            if 'heatmap' in plot_types:
                self._save_heatmap(numeric_metrics, plots_dir)
                
            logger.info(f"Plots saved to {plots_dir}")
        except Exception as e:
            logger.error(f"Failed to save plots: {str(e)}")
            
    def _save_line_plots(self, metrics: Dict[str, Dict], plots_dir: Path) -> None:
        """Save line plots of metrics."""
        for name, data in metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(data['steps'], data['values'], marker='o', linewidth=2)
            plt.title(f'{name} over Time')
            plt.xlabel('Step' if any(step is not None for step in data['steps']) else 'Index')
            plt.ylabel(name)
            plt.grid(True, alpha=0.3)
            
            # Save plot
            safe_name = "".join(c for c in name if c.isalnum() or c in "._- ").rstrip()
            filepath = plots_dir / f"{safe_name}_line.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
    def _save_heatmap(self, metrics: Dict[str, Dict], plots_dir: Path) -> None:
        """Save heatmap of metric correlations."""
        # Create DataFrame
        df_data = {}
        max_len = 0
        
        for name, data in metrics.items():
            df_data[name] = data['values']
            max_len = max(max_len, len(data['values']))
            
        # Pad sequences to same length
        for name in df_data:
            if len(df_data[name]) < max_len:
                df_data[name].extend([np.nan] * (max_len - len(df_data[name])))
                
        df = pd.DataFrame(df_data)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Metric Correlation Heatmap')
        
        # Save plot
        filepath = plots_dir / "metrics_correlation_heatmap.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'status': self.metadata.get('status', 'unknown'),
            'config': self.config,
            'metrics_summary': {},
            'metadata': self.metadata
        }
        
        # Add metrics summary
        for name, values in self.metrics.items():
            numeric_values = [entry['value'] for entry in values 
                            if isinstance(entry['value'], (int, float))]
            if numeric_values:
                summary['metrics_summary'][name] = {
                    'last_value': numeric_values[-1] if numeric_values else None,
                    'min_value': min(numeric_values) if numeric_values else None,
                    'max_value': max(numeric_values) if numeric_values else None,
                    'mean_value': np.mean(numeric_values) if numeric_values else None,
                    'count': len(numeric_values)
                }
                
        return summary
        
    def export_to_csv(self) -> None:
        """Export metrics to CSV files."""
        csv_dir = self.run_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Export individual metrics
        for name, values in self.metrics.items():
            if values:
                df = pd.DataFrame(values)
                safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
                filepath = csv_dir / f"{safe_name}.csv"
                df.to_csv(filepath, index=False)
                
        # Export combined metrics
        if self.metrics:
            combined_data = defaultdict(list)
            for name, values in self.metrics.items():
                for i, entry in enumerate(values):
                    combined_data['step'].append(entry.get('step', i))
                    combined_data['timestamp'].append(entry['timestamp'])
                    combined_data[name].append(entry['value'])
                    
            combined_df = pd.DataFrame(combined_data)
            combined_filepath = csv_dir / "all_metrics.csv"
            combined_df.to_csv(combined_filepath, index=False)
            
        logger.info(f"Metrics exported to CSV in {csv_dir}")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.update_status('completed' if exc_type is None else 'failed')
        self.save_results()
        if exc_type is not None:
            logger.error(f"Experiment failed with exception: {exc_val}")

# Global result tracker instance
_global_tracker: Optional[ResultTracker] = None

def initialize_tracker(experiment_name: str, base_dir: str = "results") -> ResultTracker:
    """
    Initialize global result tracker.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for storing results
        
    Returns:
        Initialized ResultTracker instance
    """
    global _global_tracker
    _global_tracker = ResultTracker(experiment_name, base_dir)
    return _global_tracker
    
def get_tracker() -> Optional[ResultTracker]:
    """
    Get global result tracker.
    
    Returns:
        Global ResultTracker instance or None if not initialized
    """
    return _global_tracker
    
def log_metric(name: str, value: Union[float, int], step: Optional[int] = None) -> None:
    """
    Log a metric to global tracker.
    
    Args:
        name: Name of the metric
        value: Metric value
        step: Training step (optional)
    """
    if _global_tracker is not None:
        _global_tracker.log_metric(name, value, step)
        
def log_metrics(metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
    """
    Log multiple metrics to global tracker.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Training step (optional)
    """
    if _global_tracker is not None:
        _global_tracker.log_metrics(metrics, step)

__all__ = [
    'ResultTracker',
    'initialize_tracker',
    'get_tracker',
    'log_metric',
    'log_metrics'
]
