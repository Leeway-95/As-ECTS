import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger, get_progress_manager

logger = get_logger(__name__)
progress_manager = get_progress_manager()


def generate_synthetic_dataset(dataset_name: str, n_train: int = 50, n_test: int = 20,
                              series_length: int = 100) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Generate synthetic dataset for testing when real data is not available
    with logging and progress tracking

    Args:
        dataset_name: Name for the synthetic dataset
        n_train: Number of training samples
        n_test: Number of test samples
        series_length: Length of each time series

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    logger.info(f"🧪 Generating synthetic dataset: {dataset_name}")
    logger.info(f"📊 Parameters: train={n_train}, test={n_test}, length={series_length}")

    # Generate synthetic time series data
    np.random.seed(hash(dataset_name) % 2**32)  # Consistent seed for same dataset name

    # Define classes
    classes = ["class_A", "class_B", "class_C"]

    def generate_series(class_label: str, n_samples: int) -> Tuple[np.ndarray, List[str]]:
        """Generate time series for a specific class with progress tracking"""
        series_list = []
        labels_list = []

        with progress_manager.console.status(f"[cyan]Generating {n_samples} samples for {class_label}...[/cyan]") as status:
            for i in range(n_samples):
                if class_label == "class_A":
                    # Sine wave with noise
                    t = np.linspace(0, 4*np.pi, series_length)
                    series = np.sin(t) + 0.1 * np.random.randn(series_length)
                elif class_label == "class_B":
                    # Cosine wave with noise
                    t = np.linspace(0, 4*np.pi, series_length)
                    series = np.cos(t) + 0.1 * np.random.randn(series_length)
                else:  # class_C
                    # Linear trend with noise
                    series = np.linspace(0, 1, series_length) + 0.1 * np.random.randn(series_length)

                series_list.append(series)
                labels_list.append(class_label)

                if i % 10 == 0 and i > 0:
                    status.update(status=f"[cyan]Generated {i+1}/{n_samples} samples for {class_label}[/cyan]")

        return np.array(series_list), labels_list

    # Generate training data
    logger.info(f"📚 Generating training data")
    train_data_list = []
    train_labels_list = []

    samples_per_class = n_train // len(classes)
    remaining_samples = n_train % len(classes)

    for i, class_label in enumerate(classes):
        # Distribute remaining samples to first classes
        class_samples = samples_per_class + (1 if i < remaining_samples else 0)
        logger.info(f"📝 Generating {class_samples} training samples for {class_label}")

        class_data, class_labels = generate_series(class_label, class_samples)
        train_data_list.append(class_data)
        train_labels_list.extend(class_labels)

    train_data = np.vstack(train_data_list)

    # Generate test data
    logger.info(f"🧪 Generating test data")
    test_data_list = []
    test_labels_list = []

    samples_per_class = n_test // len(classes)
    remaining_samples = n_test % len(classes)

    for i, class_label in enumerate(classes):
        # Distribute remaining samples to first classes
        class_samples = samples_per_class + (1 if i < remaining_samples else 0)
        logger.info(f"📝 Generating {class_samples} test samples for {class_label}")

        class_data, class_labels = generate_series(class_label, class_samples)
        test_data_list.append(class_data)
        test_labels_list.extend(class_labels)

    test_data = np.vstack(test_data_list)

    logger.info(f"✅ Generated synthetic dataset: train={len(train_data)}, test={len(test_data)} samples")
    logger.info(f"📊 Class distribution - Train: {dict(zip(*np.unique(train_labels_list, return_counts=True)))}")
    logger.info(f"📊 Class distribution - Test: {dict(zip(*np.unique(test_labels_list, return_counts=True)))}")

    return train_data, train_labels_list, test_data, test_labels_list


def load_dataset_with_fallback(dataset_name: str, use_synthetic: bool = True,
                              synthetic_params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Load dataset with fallback to synthetic data if real data is not available

    Args:
        dataset_name: Name of the dataset to load
        use_synthetic: Whether to use synthetic data as fallback
        synthetic_params: Parameters for synthetic data generation

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    logger.info(f"📂 Attempting to load dataset: {dataset_name}")

    # Default synthetic parameters
    if synthetic_params is None:
        synthetic_params = {
            "n_train": 50,
            "n_test": 20,
            "series_length": 100
        }

    try:
        # Try to load real dataset
        logger.info(f"🔍 Trying to load real UCR dataset: {dataset_name}")
        from utils.dataset_loader import load_ucr_dataset

        train_data, train_labels, test_data, test_labels = load_ucr_dataset(dataset_name)

        logger.info(f"✅ Successfully loaded real dataset: {dataset_name}")
        logger.info(f"📊 Dataset stats - Train: {len(train_data)}, Test: {len(test_data)}")

        return train_data, train_labels, test_data, test_labels

    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"⚠️ Could not load real dataset {dataset_name}: {e}")

        if use_synthetic:
            logger.info(f"🧪 Falling back to synthetic data generation")
            return generate_synthetic_dataset(dataset_name, **synthetic_params)
        else:
            logger.error(f"❌ Dataset {dataset_name} not available and synthetic fallback disabled")
            raise ValueError(f"Dataset {dataset_name} not available")

    except Exception as e:
        logger.error(f"❌ Unexpected error loading dataset {dataset_name}: {e}")

        if use_synthetic:
            logger.info(f"🧪 Using synthetic data due to unexpected error")
            return generate_synthetic_dataset(dataset_name, **synthetic_params)
        else:
            raise


def save_dataset_info(dataset_name: str, train_data: np.ndarray, train_labels: List[str],
                     test_data: np.ndarray, test_labels: List[str],
                     output_dir: str = "./dataset_info") -> str:
    """
    Save dataset information and statistics to a file

    Args:
        dataset_name: Name of the dataset
        train_data: Training data
        train_labels: Training labels
        test_data: Test data
        test_labels: Test labels
        output_dir: Directory to save dataset info

    Returns:
        Path to the saved info file
    """
    logger.info(f"📊 Saving dataset information for {dataset_name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    train_stats = _calculate_dataset_stats(train_data, train_labels, "training")
    test_stats = _calculate_dataset_stats(test_data, test_labels, "test")

    # Create info dictionary
    dataset_info = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "training": train_stats,
        "test": test_stats,
        "overall": {
            "total_samples": train_stats["num_samples"] + test_stats["num_samples"],
            "num_classes": len(set(train_labels + test_labels)),
            "series_length": train_data.shape[1] if len(train_data.shape) > 1 else None
        }
    }

    # Save to file
    info_file = output_path / f"{dataset_name}_info.json"

    import json
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2, default=str)

    logger.info(f"✅ Dataset info saved to {info_file}")

    return str(info_file)


def _calculate_dataset_stats(data: np.ndarray, labels: List[str], split_name: str) -> Dict[str, Any]:
    """Calculate statistics for a dataset split"""
    if len(data) == 0:
        return {"num_samples": 0, "num_classes": 0}

    # Basic statistics
    num_samples = len(data)
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    # Data shape statistics
    if len(data.shape) > 1:
        series_length = data.shape[1]
        min_length = series_length
        max_length = series_length
        avg_length = series_length
    else:
        series_length = data.shape[0] if hasattr(data, 'shape') else len(data)
        min_length = series_length
        max_length = series_length
        avg_length = series_length

    # Value statistics
    all_values = data.flatten() if hasattr(data, 'flatten') else data
    min_value = float(np.min(all_values))
    max_value = float(np.max(all_values))
    mean_value = float(np.mean(all_values))
    std_value = float(np.std(all_values))

    stats = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "series_length": series_length,
        "min_length": min_length,
        "max_length": max_length,
        "avg_length": avg_length,
        "min_value": min_value,
        "max_value": max_value,
        "mean_value": mean_value,
        "std_value": std_value,
        "class_distribution": dict(zip(unique_labels.tolist(), counts.tolist()))
    }

    logger.info(f"📊 {split_name.capitalize()} stats: {num_samples} samples, {num_classes} classes")

    return stats


def validate_dataset(train_data: np.ndarray, train_labels: List[str],
                    test_data: np.ndarray, test_labels: List[str]) -> Dict[str, Any]:
    """
    Validate dataset consistency and integrity

    Args:
        train_data: Training data
        train_labels: Training labels
        test_data: Test data
        test_labels: Test labels

    Returns:
        Validation results dictionary
    """
    logger.info("🔍 Validating dataset integrity")

    validation_results = {
        "status": "passed",
        "issues": [],
        "warnings": []
    }

    try:
        # Check data consistency
        if len(train_data) != len(train_labels):
            validation_results["issues"].append(
                f"Training data length mismatch: data={len(train_data)}, labels={len(train_labels)}"
            )

        if len(test_data) != len(test_labels):
            validation_results["issues"].append(
                f"Test data length mismatch: data={len(test_data)}, labels={len(test_labels)}"
            )

        # Check for empty datasets
        if len(train_data) == 0:
            validation_results["issues"].append("Training dataset is empty")

        if len(test_data) == 0:
            validation_results["issues"].append("Test dataset is empty")

        # Check data shapes
        if len(train_data) > 0:
            train_shape = train_data[0].shape if hasattr(train_data[0], 'shape') else (len(train_data[0]),)
            for i, sample in enumerate(train_data[1:], 1):
                sample_shape = sample.shape if hasattr(sample, 'shape') else (len(sample),)
                if sample_shape != train_shape:
                    validation_results["warnings"].append(
                        f"Training sample {i} has different shape: expected {train_shape}, got {sample_shape}"
                    )

        if len(test_data) > 0:
            test_shape = test_data[0].shape if hasattr(test_data[0], 'shape') else (len(test_data[0]),)
            for i, sample in enumerate(test_data[1:], 1):
                sample_shape = sample.shape if hasattr(sample, 'shape') else (len(sample),)
                if sample_shape != test_shape:
                    validation_results["warnings"].append(
                        f"Test sample {i} has different shape: expected {test_shape}, got {sample_shape}"
                    )

        # Check label consistency
        train_labels_set = set(train_labels)
        test_labels_set = set(test_labels)

        labels_in_test_not_train = test_labels_set - train_labels_set
        if labels_in_test_not_train:
            validation_results["warnings"].append(
                f"Labels in test but not in training: {labels_in_test_not_train}"
            )

        # Determine final status
        if validation_results["issues"]:
            validation_results["status"] = "failed"
        elif validation_results["warnings"]:
            validation_results["status"] = "warnings"

        logger.info(f"✅ Dataset validation completed: {validation_results['status']}")

        if validation_results["issues"]:
            logger.error(f"❌ Issues found: {validation_results['issues']}")

        if validation_results["warnings"]:
            logger.warning(f"⚠️ Warnings found: {validation_results['warnings']}")

    except Exception as e:
        logger.error(f"❌ Error during dataset validation: {e}")
        validation_results["status"] = "error"
        validation_results["issues"].append(f"Validation error: {str(e)}")

    return validation_results


def split_dataset(data: np.ndarray, labels: List[str], train_ratio: float = 0.8,
                 random_state: Optional[int] = None) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Split dataset into training and test sets

    Args:
        data: Complete dataset
        labels: Complete labels
        train_ratio: Ratio of training data (0-1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    logger.info(f"✂️ Splitting dataset with train ratio: {train_ratio}")

    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle indices
    indices = np.random.permutation(len(data))
    train_size = int(len(data) * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Split data
    train_data = data[train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = data[test_indices]
    test_labels = [labels[i] for i in test_indices]

    logger.info(f"✅ Dataset split completed: train={len(train_data)}, test={len(test_data)}")

    return train_data, train_labels, test_data, test_labels