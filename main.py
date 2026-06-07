import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from utils.system_core import AsEctsSystem, create_ects_system
from utils.dataset_utils import load_dataset_with_fallback, validate_dataset
from utils.logger import get_logger
from configs.config import LOGGING_CONFIG, DATASET_SELECTION_CONFIG

logger = get_logger(__name__)


def get_available_datasets(datasets_dir: str = "./datasets/UCR") -> List[str]:
    """Get list of available UCR datasets"""
    try:
        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            logger.warning(f"Datasets directory {datasets_dir} not found")
            return []
        
        # Look for dataset directories
        dataset_dirs = [d for d in datasets_path.iterdir() if d.is_dir()]
        dataset_names = sorted([d.name for d in dataset_dirs])
        
        logger.info(f"Found {len(dataset_names)} available datasets")
        return dataset_names
        
    except Exception as e:
        logger.error(f"Error discovering datasets: {e}")
        return []


def get_datasets_from_config(dataset_selection_config: Dict[str, Any]) -> List[str]:
    """Get datasets based on configuration settings"""
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        return []
    
    # Get configuration values
    specific_datasets = dataset_selection_config.get("datasets", None)
    exclude_datasets = dataset_selection_config.get("exclude_datasets", [])
    max_datasets = dataset_selection_config.get("max_datasets", None)
    dataset_filter = dataset_selection_config.get("dataset_filter", None)
    
    # Start with specific datasets if provided, otherwise all available
    if specific_datasets:
        dataset_names = [d for d in specific_datasets if d in available_datasets]
        if not dataset_names:
            logger.warning("No specified datasets found in available datasets, using all available")
            dataset_names = available_datasets.copy()
    else:
        dataset_names = available_datasets.copy()
    
    # Apply exclusions
    if exclude_datasets:
        original_count = len(dataset_names)
        dataset_names = [d for d in dataset_names if d not in exclude_datasets]
        excluded_count = original_count - len(dataset_names)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} datasets based on configuration")
    
    # Apply filter pattern
    if dataset_filter:
        try:
            import re
            pattern = re.compile(dataset_filter)
            dataset_names = [d for d in dataset_names if pattern.search(d)]
            logger.info(f"Applied dataset filter '{dataset_filter}', {len(dataset_names)} datasets remain")
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{dataset_filter}': {e}")
    
    # Apply max limit
    if max_datasets is not None and len(dataset_names) > max_datasets:
        dataset_names = dataset_names[:max_datasets]
        logger.info(f"Limited to {max_datasets} datasets based on configuration")
    
    return dataset_names


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="As-ECTS: Adaptive Shapelet Learning for Early Classification of Streaming Time Series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all available datasets
  python main.py --all
  
  # Process specific datasets
  python main.py --datasets Adiac CricketX CricketY
  
  # Process with custom configuration
  python main.py --datasets Adiac --config custom_config.json
  
  # Generate visualizations only
  python main.py --datasets Adiac --visualize-only
  
  # Save results to specific directory
  python main.py --datasets Adiac --output-dir ./my_results
        """
    )
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group(required=False)
    dataset_group.add_argument(
        "--all", 
        action="store_true",
        help="Process all available UCR datasets"
    )
    dataset_group.add_argument(
        "--datasets", 
        nargs="+", 
        type=str,
        help="List of specific datasets to process"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to custom configuration file (JSON format)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    parser.add_argument(
        "--save-state", 
        action="store_true",
        help="Save system state after processing"
    )
    
    parser.add_argument(
        "--load-state", 
        type=str, 
        default=None,
        help="Load system state from file before processing"
    )
    
    # Processing options
    parser.add_argument(
        "--visualize-only", 
        action="store_true",
        help="Only generate visualizations (skip training/evaluation)"
    )
    
    parser.add_argument(
        "--skip-visualization", 
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--max-datasets", 
        type=int, 
        default=None,
        help="Maximum number of datasets to process (for testing)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress bars and detailed output"
    )
    
    return parser.parse_args()


def load_configuration(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file"""
    config = {}
    
    # Start with default dataset selection config if no config file provided
    if config_path is None:
        logger.info("Using default configuration")
        config = {
            "dataset_selection": DATASET_SELECTION_CONFIG.copy()
        }
        return config
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file {config_path} not found")
            # Return default config even if file not found
            config = {
                "dataset_selection": DATASET_SELECTION_CONFIG.copy()
            }
            return config
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Ensure dataset_selection config exists with defaults
        if "dataset_selection" not in config:
            config["dataset_selection"] = DATASET_SELECTION_CONFIG.copy()
        else:
            # Merge with defaults for missing keys
            for key, value in DATASET_SELECTION_CONFIG.items():
                if key not in config["dataset_selection"]:
                    config["dataset_selection"][key] = value
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        # Return default config on JSON error
        config = {
            "dataset_selection": DATASET_SELECTION_CONFIG.copy()
        }
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default config on any error
        config = {
            "dataset_selection": DATASET_SELECTION_CONFIG.copy()
        }
        return config


def _deep_merge_config(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge user configuration with default configuration"""
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            # Override or add new key
            merged[key] = value
    
    return merged


def load_configuration(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file with complete default config"""
    from configs.config import (
        MATRIX_CONFIG, FOREST_CONFIG, EARLY_CONFIG,
        LOGGING_CONFIG, PROGRESS_CONFIG, VISUALIZATION_CONFIG,
        DATASET_CONFIG, DATASET_SELECTION_CONFIG
    )
    
    # Start with complete default configuration
    default_config = {
        "matrix": MATRIX_CONFIG.copy(),
        "forest": FOREST_CONFIG.copy(),
        "early": EARLY_CONFIG.copy(),
        "logging": LOGGING_CONFIG.copy(),
        "progress": PROGRESS_CONFIG.copy(),
        "visualization": VISUALIZATION_CONFIG.copy(),
        "dataset": DATASET_CONFIG.copy(),
        "dataset_selection": DATASET_SELECTION_CONFIG.copy()
    }
    
    # If no config file provided, return default config
    if config_path is None:
        logger.info("Using default configuration")
        return default_config
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file {config_path} not found")
            return default_config
        
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Deep merge user config with defaults
        config = _deep_merge_config(default_config, user_config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return default_config


def train_and_evaluate_on_datasets(dataset_names: List[str], system: AsEctsSystem) -> Dict[str, Any]:
    """
    Train and evaluate the As-ECTS system on multiple datasets with progress tracking and error handling
    
    Args:
        dataset_names: List of dataset names to process
        system: AsEctsSystem instance
        
    Returns:
        Dictionary containing training and evaluation results for all datasets
    """
    logger.info(f"📊 Starting training and evaluation on {len(dataset_names)} datasets")

    
    results = {
        "dataset_results": {},
        "overall_statistics": {},
        "total_datasets": len(dataset_names),
        "successful_datasets": 0,
        "failed_datasets": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    all_accuracies = []
    all_earliness_scores = []
    all_hm_scores = []
    
    # Process each dataset with progress tracking
    for dataset_idx, dataset_name in enumerate(dataset_names):
        logger.info(f"🔍 Processing dataset {dataset_idx + 1}/{len(dataset_names)}: {dataset_name}")
        

        # Load dataset with fallback to synthetic data
        logger.info(f"📂 Loading dataset: {dataset_name}")
        train_data, train_labels, test_data, test_labels = load_dataset_with_fallback(dataset_name)

        # Validate dataset
        validation_result = validate_dataset(train_data, train_labels, test_data, test_labels)
        if validation_result["status"] != "passed":
            logger.warning(f"⚠️ Dataset validation issues: {validation_result['issues']}")

        logger.info(f"✅ Dataset loaded: {len(train_data)} train, {len(test_data)} test samples")

        # Training phase with progress indication
        logger.info(f"📚 Starting training phase for {dataset_name}")
        training_results = system.train_on_dataset(dataset_name, train_data, train_labels)

        if training_results["status"] == "failed":
            logger.error(f"❌ Training failed for {dataset_name}: {training_results.get('error', 'Unknown error')}")
            results["dataset_results"][dataset_name] = {
                "error": training_results.get("error", "Training failed"),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            results["failed_datasets"] += 1
            continue

        logger.info(f"🎉 Training completed for {dataset_name}")
        logger.info(f"📊 Training results: {training_results}")

        # Evaluation phase
        logger.info(f"🔍 Starting evaluation for {dataset_name}")
        evaluation_results = system.evaluate_on_dataset(dataset_name, test_data, test_labels)

        if evaluation_results["status"] == "failed":
            logger.error(f"❌ Evaluation failed for {dataset_name}: {evaluation_results.get('error', 'Unknown error')}")
            results["dataset_results"][dataset_name] = {
                "training": training_results,
                "error": evaluation_results.get("error", "Evaluation failed"),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            results["failed_datasets"] += 1
            continue

        logger.info(f"🎉 Evaluation completed for {dataset_name}")
        logger.info(f"📊 Evaluation results: accuracy={evaluation_results['accuracy']:.4f}, "
                   f"earliness={evaluation_results['avg_earliness']:.4f}, "
                   f"hm_score={evaluation_results['hm_score']:.4f}")

        # Collect metrics for overall statistics
        all_accuracies.append(evaluation_results["accuracy"])
        all_earliness_scores.append(evaluation_results["avg_earliness"])
        all_hm_scores.append(evaluation_results["hm_score"])

        # Generate visualizations if enabled
        visualization_files = []
        try:
            logger.info(f"🎨 Generating visualizations for {dataset_name}")

            # Matrix evolution visualizations
            matrix_viz_files = system.visualize_matrix_evolution(dataset_name)
            visualization_files.extend(matrix_viz_files)
            logger.info(f"✅ Generated {len(matrix_viz_files)} matrix visualizations")

            # Tree visualizations using pybaobab
            logger.info(f"🌲 Generating tree visualizations for {dataset_name}")
            tree_viz_files = system.visualize_top_trees(dataset_name, top_k=3)
            visualization_files.extend(tree_viz_files)
            logger.info(f"✅ Generated {len(tree_viz_files)} tree visualizations")

            logger.info(f"🎉 Total visualizations generated: {len(visualization_files)}")

        except Exception as e:
            logger.warning(f"⚠️ Visualization generation failed: {e}")

        # Store complete results for this dataset
        results["dataset_results"][dataset_name] = {
            "training": training_results,
            "evaluation": evaluation_results,
            "visualizations": visualization_files,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }

        results["successful_datasets"] += 1
        logger.info(f"✅ Successfully processed {dataset_name}")

        
    # Calculate overall statistics
    if all_accuracies:
        results["overall_statistics"] = {
            "mean_accuracy": float(np.mean(all_accuracies)),
            "std_accuracy": float(np.std(all_accuracies)),
            "min_accuracy": float(np.min(all_accuracies)),
            "max_accuracy": float(np.max(all_accuracies)),
            "mean_earliness": float(np.mean(all_earliness_scores)),
            "std_earliness": float(np.std(all_earliness_scores)),
            "mean_hm_score": float(np.mean(all_hm_scores)),
            "std_hm_score": float(np.std(all_hm_scores)),
            "total_datasets_processed": len(all_accuracies)
        }
        
        logger.info(f"📊 Overall statistics calculated:")
        logger.info(f"   Mean accuracy: {results['overall_statistics']['mean_accuracy']:.4f} "
                   f"(±{results['overall_statistics']['std_accuracy']:.4f})")
        logger.info(f"   Mean earliness: {results['overall_statistics']['mean_earliness']:.4f} "
                   f"(±{results['overall_statistics']['std_earliness']:.4f})")
        logger.info(f"   Mean HM score: {results['overall_statistics']['mean_hm_score']:.4f} "
                   f"(±{results['overall_statistics']['std_hm_score']:.4f})")
    
    logger.info(f"🎉 Training and evaluation completed!")
    logger.info(f"   Successful: {results['successful_datasets']}/{results['total_datasets']} datasets")
    logger.info(f"   Failed: {results['failed_datasets']}/{results['total_datasets']} datasets")
    
    return results


def setup_logging(log_level: str, quiet: bool):
    """Setup logging configuration"""
    LOGGING_CONFIG["level"] = log_level
    LOGGING_CONFIG["console_level"] = "WARNING" if quiet else log_level
    
    logger.info(f"Logging configured: level={log_level}, quiet={quiet}")


def create_output_structure(output_dir: str) -> Path:
    """Create output directory structure"""
    output_path = Path(output_dir)
    
    # Create main directories
    (output_path / "results").mkdir(parents=True, exist_ok=True)
    (output_path / "logs").mkdir(parents=True, exist_ok=True)
    (output_path / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_path / "matrix_logs").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output structure at {output_path}")
    return output_path


def process_datasets(dataset_names: List[str], system_config: Dict[str, Any], 
                    output_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Process the selected datasets"""
    
    logger.info(f"Starting processing of {len(dataset_names)} datasets")
    system = create_ects_system(system_config)
    if args.load_state:
        try:
            system.load_system_state(args.load_state)
            logger.info(f"Loaded system state from {args.load_state}")
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "datasets_processed": [],
        "overall_statistics": {},
        "system_statistics": {},
        "errors": []
    }
    
    if args.visualize_only:
        logger.info("Running in visualization-only mode")
        
        for dataset_name in dataset_names:
            try:
                viz_files = system.visualize_matrix_evolution(dataset_name)
                results["datasets_processed"].append({
                    "name": dataset_name,
                    "status": "visualization_only",
                    "visualizations": viz_files
                })
                
            except Exception as e:
                logger.error(f"Error visualizing {dataset_name}: {e}")
                results["errors"].append({"dataset": dataset_name, "error": str(e)})
    
    else:
        dataset_results = train_and_evaluate_on_datasets(dataset_names, system)

        for dataset_name, dataset_result in dataset_results["dataset_results"].items():
            if "error" in dataset_result:
                results["errors"].append({"dataset": dataset_name, "error": dataset_result["error"]})
            else:
                results["datasets_processed"].append({
                    "name": dataset_name,
                    "status": "completed",
                    "training": dataset_result.get("training", {}),
                    "evaluation": dataset_result.get("evaluation", {}),
                    "visualizations": dataset_result.get("visualizations", [])
                })
        
        results["overall_statistics"] = dataset_results.get("overall_statistics", {})
        results["system_statistics"] = system.get_system_statistics()

    if args.save_state:
        state_file = Path(output_dir) / "system_state.pkl"
        try:
            system.save_system_state(str(state_file))
            logger.info(f"Saved system state to {state_file}")
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    return results


def save_results(results: Dict[str, Any], output_dir: str):
    """Save processing results to files"""
    output_path = Path(output_dir)
    
    try:
        # Save detailed results as JSON
        results_file = output_path / "results" / "processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary as text
        summary_file = output_path / "results" / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("As-ECTS Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {results['timestamp']}\n")
            f.write(f"Datasets Processed: {len(results['datasets_processed'])}\n")
            f.write(f"Errors: {len(results['errors'])}\n\n")
            
            if results['overall_statistics']:
                f.write("Overall Statistics:\n")
                for key, value in results['overall_statistics'].items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            f.write("Dataset Results:\n")
            for dataset_result in results['datasets_processed']:
                f.write(f"  {dataset_result['name']}: {dataset_result['status']}\n")
                if 'evaluation' in dataset_result and dataset_result['evaluation']:
                    eval_data = dataset_result['evaluation']
                    f.write(f"    Accuracy: {eval_data.get('accuracy', 0):.4f}\n")
                    f.write(f"    Earliness: {eval_data.get('earliness', 0):.4f}\n")
                    f.write(f"    HM Score: {eval_data.get('hm_score', 0):.4f}\n")
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Save error log if there were errors
        if results['errors']:
            error_file = output_path / "results" / "errors.txt"
            with open(error_file, 'w') as f:
                f.write("Processing Errors\n")
                f.write("=" * 30 + "\n\n")
                for error in results['errors']:
                    f.write(f"Dataset: {error['dataset']}\n")
                    f.write(f"Error: {error['error']}\n")
                    f.write("-" * 30 + "\n")
            
            logger.info(f"Saved error log to {error_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def print_welcome_message():
    """Print welcome message"""
    welcome_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    As-ECTS: Adaptive Shapelet Learning                       ║
║              for Early Classification of Streaming Time Series               ║
║                                                                              ║
║  A comprehensive implementation of the As-ECTS algorithm with:               ║
║  • Shapelet similarity matrix with incremental updates                       ║
║  • Attention-enhanced evaluation using PerMax normalization                  ║
║  • Random forest of shapelet decision trees (PDN/SDN/LDN)                    ║
║  • Early shapelet caching matching for fast classification                   ║
║  • Collaborative training with distribution change detection                 ║
║  • Comprehensive visualization and matrix evolution tracking                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(welcome_text)


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Print welcome message
    if not args.quiet:
        print_welcome_message()
    
    # Setup logging
    setup_logging(args.log_level, args.quiet)
    
    logger.info("Starting As-ECTS system")
    
    try:
        # Load configuration
        system_config = load_configuration(args.config)
        
        # Create output structure
        output_dir = create_output_structure(args.output_dir)
        
        # Get datasets to process
        if args.all:
            dataset_names = get_available_datasets()
            if args.max_datasets:
                dataset_names = dataset_names[:args.max_datasets]
        elif args.datasets:
            dataset_names = args.datasets
        else:
            # Use configuration file dataset selection
            dataset_selection_config = system_config.get("dataset_selection", DATASET_SELECTION_CONFIG)
            dataset_names = get_datasets_from_config(dataset_selection_config)
        
        # Validate dataset names
        available_datasets = get_available_datasets()
        if dataset_names:
            invalid_datasets = [d for d in dataset_names if d not in available_datasets]
            if invalid_datasets:
                logger.warning(f"Invalid dataset names: {invalid_datasets}")
                dataset_names = [d for d in dataset_names if d in available_datasets]
        else:
            # If no datasets specified, use all available datasets (default behavior)
            dataset_names = available_datasets
        
        if not dataset_names:
            logger.error("No valid datasets to process")
            return 1
        
        logger.info(f"Processing {len(dataset_names)} datasets: {dataset_names[:5]}...")
        
        # Process datasets
        results = process_datasets(dataset_names, system_config, str(output_dir), args)
        
        # Save results
        save_results(results, str(output_dir))
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("PROCESSING COMPLETED")
            print("="*60)
            print(f"Datasets processed: {len(results['datasets_processed'])}")
            print(f"Errors encountered: {len(results['errors'])}")
            
            if results['overall_statistics']:
                print(f"Mean accuracy: {results['overall_statistics'].get('mean_accuracy', 0):.4f}")
                print(f"Mean earliness: {results['overall_statistics'].get('mean_earliness', 0):.4f}")
            
            print(f"Results saved to: {output_dir}")
            print("="*60)
        
        logger.info("As-ECTS processing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)