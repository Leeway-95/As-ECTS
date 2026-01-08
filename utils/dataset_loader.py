import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from utils.logger import get_logger

logger = get_logger(__name__)

def load_ucr_dataset(dataset_name: str, datasets_dir: str = "./datasets/UCR") -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """ Load UCR dataset from files
    
    Args:
        dataset_name: Name of the dataset
        datasets_dir: Directory containing datasets
    
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    try:
        datasets_path = Path(datasets_dir)
        dataset_path = datasets_path / dataset_name
        
        if not dataset_path.exists():
            logger.error(f"Dataset {dataset_name} not found at {dataset_path}")
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        # Look for train and test files
        train_file = None
        test_file = None
        for file in dataset_path.iterdir():
            if file.suffix in ['.tsv', '.csv']:
                if 'TRAIN' in file.name.upper():
                    train_file = file
                elif 'TEST' in file.name.upper():
                    test_file = file
        
        if not train_file or not test_file:
            logger.error(f"Could not find train/test files for dataset {dataset_name}")
            raise FileNotFoundError(f"Missing train/test files for {dataset_name}")
        
        # Load training data
        logger.info(f"Loading training data from {train_file}")
        train_df = pd.read_csv(train_file, sep='\t' if train_file.suffix == '.tsv' else ',', header=None)
        
        # First column is label, rest is time series data
        train_labels = train_df.iloc[:, 0].astype(str).tolist()
        train_data = train_df.iloc[:, 1:].values
        
        # Load test data
        logger.info(f"Loading test data from {test_file}")
        test_df = pd.read_csv(test_file, sep='\t' if test_file.suffix == '.tsv' else ',', header=None)
        
        # First column is label, rest is time series data
        test_labels = test_df.iloc[:, 0].astype(str).tolist()
        test_data = test_df.iloc[:, 1:].values
        
        logger.info(f"Loaded dataset {dataset_name}: train={len(train_data)} samples, test={len(test_data)} samples")
        return train_data, train_labels, test_data, test_labels
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise