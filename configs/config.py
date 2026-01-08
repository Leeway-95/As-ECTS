from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "UCR"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
MATRIX_LOGS_DIR = RESULTS_DIR / "matrix_npz"

# Similarity matrix parameters
MATRIX_CONFIG = {
    "distance_threshold": 0.1,
    "similarity_threshold": 0.9,
    "attention_threshold": 0.8,
    "permax_scale": 1.0,
    "matrix_size": 100,
    "auto_expand": True,
    "expand_factor": 1.5,
    "max_matrix_size": 2000,
    "matrix_logs_dir": str(MATRIX_LOGS_DIR),
}

# Forest parameters
FOREST_CONFIG = {
    "n_trees": 100,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "tree_score_threshold": 0.85,
    "consistency_threshold": 0.75,
    "info_gain_threshold": 0.05,
}

# Early classification parameters
EARLY_CONFIG = {
    "early_match_threshold": 0.85,
    "max_lookback": 10,
    "min_confidence": 0.7,
    "cache_size": 500,
    "enable_adaptive_threshold": True,
    "length_tolerance": 0.2,
    "adaptive_threshold_range": [0.75, 0.95],
    "quality_weightings": {
        "success_rate": 0.35,
        "confidence": 0.25,
        "discriminative_score": 0.25,
        "recency": 0.1,
        "label_match": 0.05
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_level": "DEBUG",
    "console_level": "INFO",
    "log_file": LOGS_DIR / "as_ects.log",
    "max_bytes": 10 * 1024 * 1024,       # 10MB
    "backup_count": 5,
    "suppress_patterns": [
        "Distribution change detected",
    ],
    "suppress_during_progress": True,
}

# Progress bar configuration
PROGRESS_CONFIG = {
    "refresh_per_second": 10,
    "expand": True,
    "transient": False,
    "console_width": 120,                # Console width for progress bar display
    "enable_live_display": True,         # Enable live display mode for single-line progress
    "suppress_output_during_progress": True,  # Suppress other output during progress
    "use_rich_live": True,               # Use Rich Live context manager for better display control
    "progress_bar_style": "default",     # Progress bar style: default, minimal, detailed
    "auto_refresh": True,                # Auto refresh progress bar
    "redirect_stdout": True,             # Redirect stdout to prevent interference
    "redirect_stderr": True,             # Redirect stderr to prevent interference
}

# Visualization parameters
VISUALIZATION_CONFIG = {
    "tree_format": "png",
    "tree_dpi": 300,
    "matrix_animation_fps": 2,
    "figure_size": (12, 8),
    "enable_image_generation": True,     # Enable image generation
    "matrix_logs_dir": str(MATRIX_LOGS_DIR),  # Use results/matrix_logs
    "output_dir": str(RESULTS_DIR / "visualizations"),  # Use results/visualizations
}

# Dataset selection configuration
DATASET_SELECTION_CONFIG = {
    "datasets": None,                    # List of specific datasets to process (None = all available)
    "exclude_datasets": [],              # List of datasets to exclude
    "max_datasets": None,                # Maximum number of datasets to process (None = no limit)
    "dataset_filter": None,              # Regex pattern to filter datasets by name
}

# Dataset processing
DATASET_CONFIG = {
    "test_size": 0.2,                    # proportion of data for testing
    "validation_size": 0.1,              # proportion of data for validation
    "min_shapelet_length": 0.1,          # minimum shapelet length as proportion
    "max_shapelet_length": 0.5,          # maximum shapelet length as proportion
    "n_shapelets": 50,                   # number of shapelets to extract
}

# Ensure directories exist
for directory in [LOGS_DIR, RESULTS_DIR, MATRIX_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)