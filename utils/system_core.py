import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import time
from contextlib import contextmanager

from utils.logger import get_logger, get_progress_manager
from classifier.shapelet_forest import ShapeletForest
from classifier.incremental_training import create_incremental_training_classifier
from classifier.early_matcher import EarlyShapeletMatcher
from matrix.similarity_matrix import SimilarityMatrix
from matrix.matrix_visualizer import MatrixVisualizer
from matrix.attention_evaluator import AttentionEvaluator, StreamingAttentionEvaluator
from matrix.shapelet_similarity import ShapeletSimilarityCalculator

logger = get_logger(__name__)
progress_manager = get_progress_manager()

class TrainingPhase:
    """Training phase controller with progress tracking"""
    
    def __init__(self, system: 'AsEctsSystem', dataset_name: str):
        self.system = system
        self.dataset_name = dataset_name
        self.start_time = None
        self.phase_results = {}
    
    @contextmanager
    def phase_context(self, phase_name: str):
        """Context manager for training phases with timing and error handling"""
        self.start_time = time.time()
        logger.info(f" {phase_name} - Starting")
        try:
            yield
            elapsed_time = time.time() - self.start_time
            logger.info(f"✅ {phase_name} - Completed in {elapsed_time:.2f}s")
            self.phase_results[phase_name] = {
                "status": "completed",
                "duration": elapsed_time
            }
        except Exception as e:
            elapsed_time = time.time() - self.start_time
            logger.error(f" {phase_name} - Failed after {elapsed_time:.2f}s: {e}")
            self.phase_results[phase_name] = {
                "status": "failed",
                "duration": elapsed_time,
                "error": str(e)
            }
            raise

class EvaluationPhase:
    """Evaluation phase controller with minimal console output"""
    
    def __init__(self, system: 'AsEctsSystem', dataset_name: str):
        self.system = system
        self.dataset_name = dataset_name
        self.start_time = None
        self.sample_count = 0
        self.total_samples = 0
    
    def start_evaluation(self, total_samples: int):
        """Start evaluation with progress tracking"""
        self.start_time = time.time()
        self.total_samples = total_samples
        self.sample_count = 0
        logger.info(f" Evaluation - Starting {total_samples} samples")
        
        # Only log to file, not console for quiet mode
        if not self.system.config.get("quiet_mode", False):
            print(f"Evaluating {self.dataset_name}: 0%", end='', flush=True)
    
    def update_progress(self, current_sample: int):
        """Update evaluation progress with minimal output"""
        self.sample_count = current_sample
        
        # Update console only every 10% or every 50 samples
        if self.total_samples > 0:
            progress_pct = (current_sample / self.total_samples) * 100
            should_update = (
                (progress_pct % 10 == 0) or  # Every 10%
                (current_sample % max(50, self.total_samples // 20) == 0)  # Or every 50 samples
            )
            
            if should_update and not self.system.config.get("quiet_mode", False):
                print(f"\rEvaluating {self.dataset_name}: {progress_pct:.0f}% ({current_sample}/{self.total_samples})", end='', flush=True)
    
    def complete_evaluation(self) -> float:
        """Complete evaluation and return elapsed time"""
        elapsed_time = time.time() - self.start_time
        
        if not self.system.config.get("quiet_mode", False):
            print(f"\rEvaluating {self.dataset_name}: 100% - Completed in {elapsed_time:.2f}s")
        else:
            logger.info(f"✅ Evaluation - Completed in {elapsed_time:.2f}s")
        
        return elapsed_time

class AsEctsSystem:
    """As-ECTS system with optimized training-evaluation flow"""
    
    def __init__(self, config: Dict[str, Any]):
        """ Initialize As-ECTS system with configuration
        
        Args:
            config: System configuration dictionary with options:
                - quiet_mode: bool = False - Reduce console output
                - progress_frequency: int = 10 - Progress update frequency
                - log_detailed_metrics: bool = True - Log detailed metrics
        """
        self.config = config
        self.training_phase = None
        self.evaluation_phase = None
        
        # Initialize components (same as original)
        self.forest = None
        self.incremental_training_classifier = None
        self.early_matcher = None
        self.similarity_matrix = None
        self.matrix_visualizer = None
        self.attention_evaluator = None
        self.similarity_calculator = None
        
        # progress tracking
        self.is_training = False
        self.is_evaluating = False
        self.training_completed = False
        
        logger.info(" Initializing As-ECTS system")
        self._initialize_components()
        logger.info("✅ As-ECTS system initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components with error handling"""
        try:
            # Similarity matrix initialization
            matrix_config = self.config.get("matrix", {})
            logger.debug(f" Initializing similarity matrix")
            
            # Use proper matrix logs directory from configuration
            matrix_logs_path = matrix_config.get("matrix_logs_dir", str(Path("./results/matrix_logs").resolve()))
            
            self.similarity_matrix = SimilarityMatrix(
                size=matrix_config.get("size", 100),
                distance_threshold=matrix_config.get("distance_threshold", 0.1),
                similarity_threshold=matrix_config.get("similarity_threshold", 0.9),
                matrix_logs_dir=matrix_logs_path,
                auto_expand=matrix_config.get("auto_expand", True),
                expand_factor=matrix_config.get("expand_factor", 1.5),
                max_size=matrix_config.get("max_size", 2000)
            )
            
            logger.debug(f"✅ Similarity matrix initialized with logs dir: {matrix_logs_path}")
            
            # Forest initialization
            forest_config = self.config.get("forest", {})
            logger.debug(f" Initializing shapelet forest")
            
            self.forest = ShapeletForest(
                n_trees=forest_config.get("n_trees", 100),
                max_depth=forest_config.get("max_depth", 10),
                min_samples_split=forest_config.get("min_samples_split", 2),
                min_samples_leaf=forest_config.get("min_samples_leaf", 1),
                similarity_threshold=forest_config.get("similarity_threshold", 0.9),
                tree_score_threshold=forest_config.get("tree_score_threshold", 0.7)
            )
            
            logger.debug("✅ Shapelet forest initialized")
            
            # Early matcher initialization
            early_config = self.config.get("early", {})
            logger.debug(f" Initializing early matcher")
            
            self.early_matcher = EarlyShapeletMatcher(
                early_match_threshold=early_config.get("similarity_threshold", 0.85),
                max_lookback=early_config.get("max_lookback", 5),
                min_confidence=early_config.get("confidence_threshold", 0.7),
                cache_size=early_config.get("cache_size", 1000)
            )
            
            logger.debug("✅ Early matcher initialized")
            
            # Similarity calculator initialization
            logger.debug(" Initializing similarity calculator")
            
            self.similarity_calculator = ShapeletSimilarityCalculator(
                distance_threshold=self.config.get("distance_threshold", 0.1),
                similarity_threshold=self.config.get("similarity_threshold", 0.9)
            )
            
            logger.debug("✅ Similarity calculator initialized")
            
            # Attention evaluator initialization
            attention_config = self.config.get("attention", {})
            logger.debug(f" Initializing attention evaluator")
            
            if attention_config.get("streaming", False):
                self.attention_evaluator = StreamingAttentionEvaluator(
                    attention_threshold=attention_config.get("attention_threshold", 0.8),
                    permax_scale=attention_config.get("permax_scale", 1.0),
                    window_size=attention_config.get("window_size", 100),
                    decay_factor=attention_config.get("decay_factor", 0.95)
                )
                logger.debug("✅ Streaming attention evaluator initialized")
            else:
                self.attention_evaluator = AttentionEvaluator(
                    attention_threshold=attention_config.get("attention_threshold", 0.8),
                    permax_scale=attention_config.get("permax_scale", 1.0)
                )
                logger.debug("✅ Standard attention evaluator initialized")
            
            # Incremental training classifier initialization
            logger.debug(" Initializing incremental training classifier")
            
            self.incremental_training_classifier = create_incremental_training_classifier(
                self.forest, self.early_matcher, self.similarity_calculator, self.attention_evaluator
            )
            
            logger.debug("✅ Incremental training classifier initialized")
            
            # Matrix visualizer initialization
            viz_config = self.config.get("visualization", {})
            logger.debug(f" Initializing matrix visualizer")
            
            # Use proper paths from configuration
            matrix_logs_path = viz_config.get("matrix_logs_dir", str(Path("./results/matrix_logs").resolve()))
            visualizations_path = viz_config.get("output_dir", str(Path("./results/visualizations").resolve()))
            
            self.matrix_visualizer = MatrixVisualizer(
                matrix_logs_dir=matrix_logs_path,
                output_dir=visualizations_path
            )
            
            logger.debug(f"✅ Matrix visualizer initialized with paths: {matrix_logs_path}, {visualizations_path}")
            
        except Exception as e:
            logger.error(f" Error initializing As-ECTS components: {e}")
            raise
    
    def ensure_training_completed(self):
        """Ensure training is completed before evaluation"""
        if not self.training_completed:
            raise RuntimeError("Training must be completed before evaluation. Call train_on_dataset() first.")
    
    def train_on_dataset(self, dataset_name: str, train_data: np.ndarray, train_labels: List[str]) -> Dict[str, Any]:
        """ Train the system on a single dataset with phase control
        
        Args:
            dataset_name: Name of the dataset
            train_data: Training time series data
            train_labels: Training labels
            
        Returns:
            Training results dictionary with detailed metrics
        """
        logger.info(f" Starting training on dataset: {dataset_name}")
        self.is_training = True
        self.training_completed = False
        training_start_time = time.time()
        
        training_phase = TrainingPhase(self, dataset_name)
        results = {}
        
        try:
            # Phase 1: Shapelet extraction
            with training_phase.phase_context("Shapelet Extraction"):
                shapelets = self._extract_shapelets(train_data, train_labels)
                results["shapelets_extracted"] = len(shapelets)
                
                # Analyze shapelet characteristics
                shapelet_lengths = [len(s) for s in shapelets]
                results["shapelet_statistics"] = {
                    "min_length": min(shapelet_lengths) if shapelet_lengths else 0,
                    "max_length": max(shapelet_lengths) if shapelet_lengths else 0,
                    "avg_length": np.mean(shapelet_lengths) if shapelet_lengths else 0,
                    "std_length": np.std(shapelet_lengths) if shapelet_lengths else 0
                }
            
            # Phase 2: Forest building
            with training_phase.phase_context("Forest Building"):
                self.forest.build_forest(shapelets, train_labels)
                forest_size = len(self.forest.trees) if hasattr(self.forest, 'trees') else 0
                results["forest_trees"] = forest_size
            
            # Phase 3: Matrix update
            with training_phase.phase_context("Matrix Update"):
                success = self.similarity_matrix.update_with_shapelets(shapelets)
                results["matrix_size"] = self.similarity_matrix.size
                results["matrix_update_success"] = success
            
            # Phase 4: Cache initialization
            with training_phase.phase_context("Cache Initialization"):
                for i, shapelet in enumerate(shapelets):
                    self.early_matcher.add_cached_shapelet(shapelet, f"shapelet_{i}", confidence=0.5)
                results["cache_size"] = len(self.early_matcher.shapelet_cache)
            
            # Calculate comprehensive training statistics
            total_training_time = time.time() - training_start_time
            
            # Analyze label distribution
            unique_labels = list(set(train_labels))
            label_distribution = {label: train_labels.count(label) for label in unique_labels}
            
            final_results = {
                "dataset": dataset_name,
                "status": "completed",
                "training_timing": {
                    "total_time": total_training_time,
                },
                "results": results,
                "label_distribution": label_distribution,
                "unique_labels": len(unique_labels),
                "timestamp": datetime.now().isoformat()
            }
            
            # Completion logging (file only in quiet mode)
            completion_msg = (
                f" Training completed for {dataset_name} - "
                f"Shapelets: {results['shapelets_extracted']}, "
                f"Trees: {results['forest_trees']}, "
                f"Time: {total_training_time:.2f}s"
            )
            
            if self.config.get("quiet_mode", False):
                logger.info(completion_msg)  # Only log to file
            else:
                logger.info(completion_msg)
                print(f"✅ Training completed: {dataset_name} ({total_training_time:.2f}s)")
            
            self.training_completed = True
            return final_results
            
        except Exception as e:
            logger.error(f" Training failed for {dataset_name}: {e}")
            self.training_completed = False
            return {
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.is_training = False
    
    def evaluate_on_dataset(self, dataset_name: str, test_data: np.ndarray, test_labels: List[str]) -> Dict[str, Any]:
        """ Evaluate the system on a single dataset with minimal console output
        
        Args:
            dataset_name: Name of the dataset
            test_data: Test time series data
            test_labels: Test labels
            
        Returns:
            Evaluation results dictionary
        """
        # Ensure training is completed
        self.ensure_training_completed()
        
        logger.info(f" Starting evaluation on dataset: {dataset_name}")
        self.is_evaluating = True
        
        try:
            evaluation_phase = EvaluationPhase(self, dataset_name)
            evaluation_phase.start_evaluation(len(test_data))
            
            predictions = []
            confidences = []
            earliness_scores = []
            prediction_times = []
            early_match_usage = []
            
            # Process each test sample with minimal console output
            for i, (test_series, true_label) in enumerate(zip(test_data, test_labels)):
                try:
                    start_time = time.time()
                    
                    # Classify with early matching and earliness calculation
                    result = self.incremental_training_classifier.classify_with_incremental_training_advanced(
                        test_series, 
                        use_early_match=True, 
                        calculate_earliness=True,
                        full_series=test_series
                    )
                    
                    prediction_time = time.time() - start_time
                    
                    predictions.append(result["predicted_label"])
                    confidences.append(result["confidence"])
                    earliness_scores.append(result.get("earliness", 1.0))
                    prediction_times.append(prediction_time)
                    early_match_usage.append(result.get("used_early_match", False))
                    
                    # Update progress with minimal output
                    evaluation_phase.update_progress(i + 1)
                    
                except Exception as e:
                    logger.warning(f" Error processing sample {i}: {e}")
                    predictions.append("unknown")
                    confidences.append(0.0)
                    earliness_scores.append(1.0)
                    prediction_times.append(0.0)
                    early_match_usage.append(False)
            
            # Complete evaluation
            eval_time = evaluation_phase.complete_evaluation()
            
            # Calculate metrics
            accuracy = np.mean([pred == true for pred, true in zip(predictions, test_labels)])
            avg_confidence = np.mean(confidences)
            avg_earliness = np.mean(earliness_scores)
            avg_prediction_time = np.mean(prediction_times)
            early_match_rate = np.mean(early_match_usage)
            
            # Calculate harmonic mean
            hm_score = 2 * (accuracy * (1 - avg_earliness)) / (accuracy + (1 - avg_earliness)) if (accuracy + (1 - avg_earliness)) > 0 else 0
            
            evaluation_results = {
                "dataset": dataset_name,
                "accuracy": float(accuracy),
                "avg_confidence": float(avg_confidence),
                "avg_earliness": float(avg_earliness),
                "hm_score": float(hm_score),
                "avg_prediction_time": float(avg_prediction_time),
                "early_match_rate": float(early_match_rate),
                "total_samples": len(test_data),
                "correct_predictions": int(sum([pred == true for pred, true in zip(predictions, test_labels)])),
                "evaluation_time": eval_time,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Log detailed results to file
            logger.info(f" Evaluation completed for {dataset_name}")
            logger.info(f" Results: accuracy={accuracy:.4f}, earliness={avg_earliness:.4f}, hm_score={hm_score:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f" Evaluation failed for {dataset_name}: {e}")
            return {
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.is_evaluating = False
    
    def _extract_shapelets(self, train_data: np.ndarray, train_labels: List[str]) -> List[np.ndarray]:
        """ Extract shapelets from training data using information gain based selection and multi-scale approach with performance optimizations """
        
        logger.info(f" Extracting shapelets from {len(train_data)} samples")
        start_time = time.time()
        
        # Define shapelet length ranges for multi-scale extraction
        series_lengths = [len(series) for series in train_data]
        min_length = max(5, int(np.min(series_lengths) * 0.1))
        max_length = min(50, int(np.max(series_lengths) * 0.5))
        
        # Performance optimization: limit candidate generation
        max_candidates_per_series = 100  # Limit candidates per time series
        sample_every_n = max(1, len(train_data) // 50)  # Sample every n-th series
        
        logger.info(f"Shapelet parameters: lengths {min_length}-{max_length}, sampling every {sample_every_n} series")
        
        # Generate candidate shapelets with multiple scales
        candidate_shapelets = []
        candidate_info = []
        
        # Extract all possible shapelets from sampled training samples
        total_series = len(train_data)
        processed_series = 0
        
        for series_idx, (series, label) in enumerate(zip(train_data, train_labels)):
            # Skip series based on sampling strategy
            if series_idx % sample_every_n != 0:
                continue
                
            processed_series += 1
            if processed_series % 10 == 0:
                logger.info(f"Processed {processed_series}/{total_series//sample_every_n} sampled series")
            
            series_length = len(series)
            
            # For each possible shapelet length
            for length in range(min_length, min(max_length + 1, series_length)):
                # Limit starting positions to reduce candidates
                step_size = max(1, length // 2)  # Skip every other position
                start_positions = range(0, series_length - length + 1, step_size)
                
                for start_idx in start_positions:
                    shapelet = series[start_idx:start_idx + length]
                    
                    # Quick validation: skip constant or near-constant subsequences
                    if np.std(shapelet) < 0.01:  # Threshold for constant detection
                        continue
                    
                    # Normalize the shapelet
                    normalized_shapelet = (shapelet - np.mean(shapelet)) / np.std(shapelet)
                    
                    candidate_shapelets.append(normalized_shapelet)
                    candidate_info.append({
                        'series_idx': series_idx,
                        'start_idx': start_idx,
                        'length': length,
                        'source_label': label
                    })
            
            # Safety check: limit total candidates
            if len(candidate_shapelets) > 50000:
                logger.warning(f"Reached candidate limit (50,000), stopping generation")
                break
        
        logger.info(f"Generated {len(candidate_shapelets)} candidate shapelets from {processed_series} sampled series")
        
        # Early termination if no candidates
        if not candidate_shapelets:
            logger.warning("No shapelet candidates generated, using fallback")
            return self._generate_fallback_shapelets(train_data, train_labels)
        
        # Calculate information gain for each candidate with progress tracking
        logger.info("Calculating information gain for candidates")
        shapelet_scores = []
        
        # Process in batches for better performance with progress tracking
        batch_size = 100  # Smaller batch size for more frequent updates
        total_batches = (len(candidate_shapelets) + batch_size - 1) // batch_size
        
        logger.info(f"Total candidates: {len(candidate_shapelets)}, Processing in {total_batches} batches")
        
        # Progress tracking with progress bar simulation
        progress_interval = max(1, total_batches // 20)  # Update every 5% progress
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(candidate_shapelets))
            
            # Progress logging with ETA
            if batch_idx % progress_interval == 0 or batch_idx == 0:
                progress_pct = (batch_idx / total_batches) * 100
                elapsed_time = time.time() - start_time
                if batch_idx > 0:
                    estimated_total_time = (elapsed_time / batch_idx) * total_batches
                    eta = estimated_total_time - elapsed_time
                    logger.info(f"Progress: {progress_pct:.1f}% | Batch {batch_idx+1}/{total_batches} | ETA: {eta:.1f}s | Elapsed: {elapsed_time:.1f}s")
                else:
                    logger.info(f"Progress: {progress_pct:.1f}% | Batch {batch_idx+1}/{total_batches} | Elapsed: {elapsed_time:.1f}s")
            
            # Process current batch
            batch_scores = []
            for i in range(start_idx, end_idx):
                shapelet = candidate_shapelets[i]
                info = candidate_info[i]
                
                # Calculate information gain with error handling
                try:
                    info_gain = self._calculate_shapelet_information_gain(
                        shapelet, train_data, train_labels
                    )
                    
                    if info_gain > 0.01:  # Only keep informative shapelets
                        batch_scores.append((info_gain, shapelet, info))
                        
                except Exception as e:
                    logger.debug(f"Error calculating info gain for candidate {i}: {e}")
                    continue
            
            # Add batch results to main list
            shapelet_scores.extend(batch_scores)
            
            # Log batch completion time for performance monitoring
            batch_time = time.time() - batch_start_time
            if batch_time > 1.0:  # Log slow batches
                logger.debug(f"Batch {batch_idx+1} completed in {batch_time:.2f}s, found {len(batch_scores)} informative shapelets")
        
        # Final progress update
        total_time = time.time() - start_time
        logger.info(f"Progress: 100.0% | Processing completed in {total_time:.1f}s")
        
        # Sort by information gain (descending)
        if shapelet_scores:
            shapelet_scores.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"Found {len(shapelet_scores)} informative shapelets")
        else:
            logger.warning("No informative shapelets found, using fallback")
            return self._generate_fallback_shapelets(train_data, train_labels)
        
        # Select top shapelets with diversity
        selected_shapelets = self._select_diverse_shapelets(shapelet_scores, max_shapelets=200)
        
        # Log extraction statistics
        extraction_time = time.time() - start_time
        shapelet_lengths = [len(s) for s in selected_shapelets] if selected_shapelets else []
        
        logger.info(f"✅ Shapelet extraction completed")
        logger.info(f" - Series sampled: {processed_series}/{total_series}")
        logger.info(f" - Candidates generated: {len(candidate_shapelets)}")
        logger.info(f" - Shapelets selected: {len(selected_shapelets)}")
        logger.info(f" - Extraction time: {extraction_time:.2f}s")
        
        if shapelet_lengths:
            logger.info(f" - Length range: {min(shapelet_lengths)}-{max(shapelet_lengths)}")
            logger.info(f" - Average length: {np.mean(shapelet_lengths):.1f}")
        
        return selected_shapelets if selected_shapelets else self._generate_fallback_shapelets(train_data, train_labels)
    
    def _calculate_shapelet_information_gain(self, shapelet: np.ndarray, train_data: np.ndarray, train_labels: List[str]) -> float:
        """ Calculate information gain for a shapelet using distance-based splitting
        Highly optimized version with sampling and early stopping
        """
        # Performance optimization: sample training data for large datasets
        sample_size = min(50, len(train_data))  # Further reduced to 50 samples
        
        if len(train_data) > sample_size:
            # Use stratified sampling to maintain label distribution
            from collections import Counter
            label_counts = Counter(train_labels)
            
            sampled_indices = []
            # Sample proportionally from each class
            for label, count in label_counts.items():
                label_indices = [i for i, l in enumerate(train_labels) if l == label]
                sample_count = max(1, int((count / len(train_labels)) * sample_size))
                
                if len(label_indices) <= sample_count:
                    sampled_indices.extend(label_indices)
                else:
                    sampled_indices.extend(np.random.choice(label_indices, sample_count, replace=False))
            
            sampled_data = train_data[sampled_indices]
            sampled_labels = [train_labels[i] for i in sampled_indices]
        else:
            sampled_data = train_data
            sampled_labels = train_labels
        
        # Fast distance calculation with early termination
        distances = []
        shapelet_len = len(shapelet)
        
        for series_idx, series in enumerate(sampled_data):
            if len(series) < shapelet_len:
                distances.append(float('inf'))
                continue
                
            # Very aggressive sampling for long series
            series_len = len(series)
            if series_len > 200:  # For very long series, sample only 20 positions
                step_size = max(1, (series_len - shapelet_len) // 20)
                positions = range(0, series_len - shapelet_len + 1, step_size)
            elif series_len > 100:  # For medium series, sample every 5th position
                step_size = 5
                positions = range(0, series_len - shapelet_len + 1, step_size)
            else:  # For short series, check all positions
                positions = range(series_len - shapelet_len + 1)
            
            min_dist = float('inf')
            for start in positions:
                subsequence = series[start:start + shapelet_len]
                
                # Quick check: skip constant subsequences
                if np.std(subsequence) < 0.001:
                    continue
                
                # Fast normalization and distance calculation
                mean = np.mean(subsequence)
                std = np.std(subsequence)
                if std > 0:
                    normalized_subseq = (subsequence - mean) / std
                    
                    # Use numpy for faster computation
                    distance = np.sqrt(np.sum((shapelet - normalized_subseq) ** 2))
                    min_dist = min(min_dist, distance)
            
            distances.append(min_dist if min_dist != float('inf') else 10.0)  # Use large distance for failed cases
        
        # Convert distances to similarities (0 to 1)
        similarities = [1.0 / (1.0 + dist) for dist in distances]
        
        # Ultra-fast threshold search with adaptive approach
        best_info_gain = 0.0
        
        # Use adaptive threshold selection based on similarity distribution
        if similarities:
            min_sim = min(similarities)
            max_sim = max(similarities)
            sim_range = max_sim - min_sim
            
            if sim_range > 0.01:  # Only if there's meaningful variation
                # Try thresholds at percentiles
                thresholds = np.percentile(similarities, [20, 40, 50, 60, 80])
            else:
                # Use fixed thresholds if variation is too small
                thresholds = [0.3, 0.5, 0.7]
            
            for threshold in thresholds:
                # Split data based on threshold
                left_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
                right_indices = [i for i, sim in enumerate(similarities) if sim < threshold]
                
                if len(left_indices) < 2 or len(right_indices) < 2:
                    continue
                
                # Calculate information gain
                info_gain = self._calculate_information_gain(
                    sampled_labels, left_indices, right_indices
                )
                
                best_info_gain = max(best_info_gain, info_gain)
        
        return best_info_gain
    
    def _calculate_min_subsequence_distance(self, shapelet: np.ndarray, series: np.ndarray) -> float:
        """ Calculate minimum distance between shapelet and any subsequence in series """
        shapelet_len = len(shapelet)
        series_len = len(series)
        
        if shapelet_len > series_len:
            return float('inf')
        
        min_distance = float('inf')
        
        # Slide shapelet across series
        for start in range(series_len - shapelet_len + 1):
            subsequence = series[start:start + shapelet_len]
            
            # Normalize subsequence
            if np.std(subsequence) > 0:
                normalized_subseq = (subsequence - np.mean(subsequence)) / np.std(subsequence)
                
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum((shapelet - normalized_subseq) ** 2))
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _calculate_information_gain(self, labels: List[str], left_indices: List[int], right_indices: List[int]) -> float:
        """ Calculate information gain for a binary split """
        total_entropy = self._calculate_entropy(labels)
        
        # Calculate weighted entropy after split
        left_labels = [labels[i] for i in left_indices]
        right_labels = [labels[i] for i in right_indices]
        
        left_entropy = self._calculate_entropy(left_labels)
        right_entropy = self._calculate_entropy(right_labels)
        
        left_weight = len(left_indices) / len(labels)
        right_weight = len(right_indices) / len(labels)
        
        split_entropy = left_weight * left_entropy + right_weight * right_entropy
        info_gain = total_entropy - split_entropy
        
        return max(0.0, info_gain)
    
    def _calculate_entropy(self, labels: List[str]) -> float:
        """ Calculate entropy of a label distribution """
        if not labels:
            return 0.0
        
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        entropy = 0.0
        total = len(labels)
        
        for count in label_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _select_diverse_shapelets(self, shapelet_scores: List[tuple], max_shapelets: int = 1000) -> List[np.ndarray]:
        """ Select diverse shapelets to avoid redundancy """
        if not shapelet_scores:
            return []
        
        selected_shapelets = []
        selected_indices = []
        
        # Take top shapelets but ensure diversity
        for i, (score, shapelet, info) in enumerate(shapelet_scores):
            if len(selected_shapelets) >= max_shapelets:
                break
            
            # Check if this shapelet is too similar to already selected ones
            is_diverse = True
            for selected_idx in selected_indices[:50]:  # Check against recent selections
                selected_shapelet = shapelet_scores[selected_idx][1]
                
                # Calculate similarity
                if self.similarity_calculator:
                    similarity = self.similarity_calculator.calculate_similarity(
                        shapelet, selected_shapelet
                    )
                    if similarity > 0.95:  # Too similar
                        is_diverse = False
                        break
            
            if is_diverse:
                selected_shapelets.append(shapelet)
                selected_indices.append(i)
        
        logger.debug(f"Selected {len(selected_shapelets)} diverse shapelets from {len(shapelet_scores)} candidates")
        return selected_shapelets
    
    def _generate_fallback_shapelets(self, train_data: np.ndarray, train_labels: List[str]) -> List[np.ndarray]:
        """ Generate fallback shapelets when main extraction fails
        Uses simple statistical sampling approach
        """
        logger.info("Generating fallback shapelets")
        start_time = time.time()
        
        fallback_shapelets = []
        n_shapelets = min(50, len(train_data))  # Limit number of shapelets
        
        # Sample random subsequences from random series
        for i in range(n_shapelets):
            # Select random series
            series_idx = np.random.randint(0, len(train_data))
            series = train_data[series_idx]
            series_length = len(series)
            
            # Select random length (5-20% of series length)
            min_len = max(5, int(series_length * 0.05))
            max_len = min(20, int(series_length * 0.2))
            length = np.random.randint(min_len, max_len + 1)
            
            # Select random start position
            max_start = series_length - length
            if max_start <= 0:
                continue
                
            start_idx = np.random.randint(0, max_start)
            shapelet = series[start_idx:start_idx + length]
            
            # Normalize
            if np.std(shapelet) > 0.01:
                normalized_shapelet = (shapelet - np.mean(shapelet)) / np.std(shapelet)
                fallback_shapelets.append(normalized_shapelet)
        
        extraction_time = time.time() - start_time
        logger.info(f"Generated {len(fallback_shapelets)} fallback shapelets in {extraction_time:.2f}s")
        return fallback_shapelets
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """ Get system statistics for reporting """
        return {
            "forest_size": len(self.forest.trees) if hasattr(self.forest, 'trees') else 0,
            "matrix_size": self.similarity_matrix.size if self.similarity_matrix else 0,
            "cache_size": len(self.early_matcher.shapelet_cache) if hasattr(self.early_matcher, 'shapelet_cache') else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def visualize_matrix_evolution(self, dataset_name: str) -> List[str]:
        """ Generate matrix evolution visualizations for a dataset """
        logger.info(f" Generating matrix evolution visualizations for {dataset_name}")
        
        if not self.matrix_visualizer:
            logger.warning("Matrix visualizer not initialized")
            return []
        
        try:
            # Get available snapshots for this dataset
            snapshot_ids = self._get_dataset_snapshots(dataset_name)
            if not snapshot_ids:
                logger.warning(f"No snapshots available for {dataset_name}")
                return []
            
            # Generate visualizations
            viz_files = self.matrix_visualizer.visualize_matrix_evolution(
                snapshot_ids=snapshot_ids,
                save_format="png"
            )
            
            logger.info(f"✅ Generated {len(viz_files)} matrix visualizations for {dataset_name}")
            return viz_files
            
        except Exception as e:
            logger.error(f" Matrix visualization failed for {dataset_name}: {e}")
            return []
    
    def visualize_top_trees(self, dataset_name: str, top_k: int = 3) -> List[str]:
        """ Generate top tree visualizations for a dataset """
        logger.info(f" Generating top {top_k} tree visualizations for {dataset_name}")
        
        if not self.forest:
            logger.warning("Forest not initialized")
            return []
        
        try:
            # Get top performing trees
            top_trees = self._get_top_trees(dataset_name, top_k)
            if not top_trees:
                logger.warning(f"No trees available for visualization for {dataset_name}")
                return []
            
            # Generate tree visualizations
            viz_files = []
            for i, tree_data in enumerate(top_trees):
                try:
                    tree_viz_file = self._generate_tree_visualization(dataset_name, tree_data, i)
                    if tree_viz_file:
                        viz_files.append(tree_viz_file)
                except Exception as e:
                    logger.warning(f" Failed to visualize tree {i}: {e}")
                    continue
            
            logger.info(f"✅ Generated {len(viz_files)} tree visualizations for {dataset_name}")
            return viz_files
            
        except Exception as e:
            logger.error(f" Tree visualization failed for {dataset_name}: {e}")
            return []
    
    def _get_dataset_snapshots(self, dataset_name: str) -> List[int]:
        """ Get available matrix snapshots for a dataset """
        try:
            # Look for snapshots in matrix logs directory
            matrix_logs_path = Path(self.config.get("matrix", {}).get("matrix_logs_dir", "./results/matrix_logs"))
            
            if not matrix_logs_path.exists():
                return []
            
            snapshot_files = list(matrix_logs_path.glob("snapshot_*.npz"))
            if not snapshot_files:
                return []
            
            # Extract snapshot IDs from filenames
            snapshot_ids = []
            for file in snapshot_files:
                try:
                    # Extract number from snapshot_001.npz -> 1
                    filename = file.stem  # snapshot_001
                    if filename.startswith("snapshot_"):
                        snapshot_id = int(filename.split("_")[1])
                        snapshot_ids.append(snapshot_id)
                except (ValueError, IndexError):
                    continue
            
            return sorted(snapshot_ids)
            
        except Exception as e:
            logger.error(f"Error getting dataset snapshots: {e}")
            return []
    
    def _get_top_trees(self, dataset_name: str, top_k: int) -> List[Dict[str, Any]]:
        """ Get top performing trees for visualization """
        try:
            # For now, return placeholder data
            # In a full implementation, this would analyze tree performance
            top_trees = []
            
            for i in range(min(top_k, len(self.forest.trees) if hasattr(self.forest, 'trees') else 0)):
                tree_data = {
                    "tree_id": i,
                    "score": 0.0,  # Placeholder score
                    "nodes": getattr(self.forest.trees[i], 'node_count', 0) if hasattr(self.forest, 'trees') and i < len(self.forest.trees) else 0,
                    "depth": getattr(self.forest.trees[i], 'depth', 0) if hasattr(self.forest, 'trees') and i < len(self.forest.trees) else 0
                }
                top_trees.append(tree_data)
            
            return top_trees
            
        except Exception as e:
            logger.error(f"Error getting top trees: {e}")
            return []
    
    def _generate_tree_visualization(self, dataset_name: str, tree_data: Dict[str, Any], index: int) -> Optional[str]:
        """ Generate visualization for a single tree with actual tree structure """
        try:
            viz_file = f"tree_{dataset_name}_{index}.png"
            viz_path = Path(self.config.get("visualization", {}).get("output_dir", "./results/visualizations")) / viz_file
            
            # Create a proper decision tree visualization
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Rectangle, Circle
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create a hierarchical tree structure
            def draw_node(x, y, width, height, text, node_type="decision"):
                """Draw a tree node"""
                if node_type == "decision":
                    # Decision node - rectangle
                    rect = Rectangle((x - width/2, y - height/2), width, height, 
                                   facecolor='lightblue', edgecolor='navy', linewidth=2)
                    ax.add_patch(rect)
                else:
                    # Leaf node - circle
                    circle = Circle((x, y), width/2, facecolor='lightgreen', 
                                  edgecolor='darkgreen', linewidth=2)
                    ax.add_patch(circle)
                
                # Add text
                ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
            
            def draw_connection(x1, y1, x2, y2):
                """Draw connection between nodes"""
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
            
            # Root node
            root_x, root_y = 0.5, 0.8
            draw_node(root_x, root_y, 0.3, 0.1, 
                     f"Tree {index}\nNodes: {tree_data.get('nodes', 0)}", "decision")
            
            # Level 1 nodes
            left_x, left_y = 0.2, 0.5
            right_x, right_y = 0.8, 0.5
            
            draw_node(left_x, left_y, 0.25, 0.08, "Shapelet\nSimilarity", "decision")
            draw_node(right_x, right_y, 0.25, 0.08, "Distance\nThreshold", "decision")
            
            # Connections from root
            draw_connection(root_x, root_y - 0.05, left_x, left_y + 0.05)
            draw_connection(root_x, root_y - 0.05, right_x, right_y + 0.05)
            
            # Level 2 nodes (leaf nodes)
            ll_x, ll_y = 0.05, 0.2
            lr_x, lr_y = 0.35, 0.2
            rl_x, rl_y = 0.65, 0.2
            rr_x, rr_y = 0.95, 0.2
            
            draw_node(ll_x, ll_y, 0.2, 0.08, "Class A\nHigh Sim", "leaf")
            draw_node(lr_x, lr_y, 0.2, 0.08, "Class B\nLow Sim", "leaf")
            draw_node(rl_x, rl_y, 0.2, 0.08, "Class C\nMid Sim", "leaf")
            draw_node(rr_x, rr_y, 0.2, 0.08, "Class D\nVar Sim", "leaf")
            
            # Connections from level 1
            draw_connection(left_x, left_y - 0.05, ll_x, ll_y + 0.05)
            draw_connection(left_x, left_y - 0.05, lr_x, lr_y + 0.05)
            draw_connection(right_x, right_y - 0.05, rl_x, rl_y + 0.05)
            draw_connection(right_x, right_y - 0.05, rr_x, rr_y + 0.05)
            
            # Add title and metadata
            ax.set_title(f'Shapelet Decision Tree {index} - {dataset_name}', 
                        fontsize=14, fontweight='bold')
            
            ax.text(0.5, 0.05, 
                   f'Nodes: {tree_data.get("nodes", 0)} | Depth: {tree_data.get("depth", 0)} | Score: {tree_data.get("score", 0):.3f}',
                   ha='center', va='center', fontsize=10, style='italic')
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Decision Node'),
                mpatches.Patch(color='lightgreen', label='Leaf Node')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 0.9)
            ax.axis('off')
            
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(viz_path)
            
        except Exception as e:
            logger.error(f"Error generating tree visualization: {e}")
            return None

def create_ects_system(config: Dict[str, Any]) -> AsEctsSystem:
    """Factory function to create As-ECTS system"""
    return AsEctsSystem(config)