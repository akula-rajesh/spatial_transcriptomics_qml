#!/usr/bin/env python3
"""
Main entry point for the spatial transcriptomics ML pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.pipeline_orchestrator import PipelineOrchestrator
from src.training.cross_validator import CrossValidator
from src.utils.directory_utils import ensure_project_structure

# Ensure project directory structure exists
ensure_project_structure(project_root)

# Configure logging (logs directory now guaranteed to exist)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'logs' / 'main.log')
    ]
)
logger = logging.getLogger(__name__)


def run_training_pipeline(config_path: str, resume_path: str = None) -> dict:
    """
    Run the full training pipeline using the orchestrator.

    Args:
        config_path: Path to configuration file
        resume_path: Optional path to a saved model checkpoint (.pth) to
                     resume training from. Weights are loaded before the
                     first training epoch.

    Returns:
        Dictionary containing pipeline results
    """
    logger.info("=" * 70)
    logger.info("STARTING SPATIAL TRANSCRIPTOMICS ML PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration: {config_path}")
    logger.info("Mode: Training")
    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
    logger.info("=" * 70)

    # Create and run pipeline orchestrator
    orchestrator = PipelineOrchestrator(config_path, resume_path=resume_path)

    # Display experiment info
    exp_info = orchestrator.get_experiment_info()
    logger.info(f"Experiment ID: {exp_info['experiment_id']}")
    logger.info(f"Experiment Directory: {exp_info['experiment_dir']}")
    logger.info("=" * 70)

    # Execute pipeline
    results = orchestrator.run_pipeline()

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return results


def run_cross_validation_pipeline(config_path: str) -> dict:
    """
    Run cross-validation pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing cross-validation results
    """
    logger.info("=" * 70)
    logger.info("STARTING CROSS-VALIDATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration: {config_path}")
    logger.info("Mode: Cross-Validation")
    logger.info("=" * 70)

    # Load config for cross-validator
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create and run cross-validator
    cross_validator = CrossValidator(config)
    results = cross_validator.run_cross_validation()

    logger.info("=" * 70)
    logger.info("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Spatial Transcriptomics ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full training pipeline
  python src/main.py --config config/pipeline_config.yaml --mode train
  
  # Run cross-validation
  python src/main.py --config config/pipeline_config.yaml --mode cross_validate
  
  # Run with verbose logging
  python src/main.py --config config/pipeline_config.yaml --mode train --verbose
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['train', 'cross_validate'],
        default='train',
        help='Execution mode: train (default) or cross_validate'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG level) logging'
    )

    parser.add_argument(
        '--resume', '-r',
        type=str,
        help='Path to a saved model checkpoint (.pth) to resume training from'
    )

    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        # Run appropriate pipeline based on mode
        # ConfigManager expects relative path from config/ directory or absolute path
        # If user provides "config/pipeline_config.yaml", we need just "pipeline_config.yaml"
        config_file = str(config_path)
        if config_file.startswith('config/'):
            config_file = config_file[7:]  # Remove "config/" prefix

        if args.mode == 'train':
            results = run_training_pipeline(config_file, resume_path=args.resume)
        elif args.mode == 'cross_validate':
            results = run_cross_validation_pipeline(config_file)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        logger.info("Pipeline execution completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
