import numpy as np
import math
import time
from typing import Dict, List, Tuple
from collections import deque

from utils.logger import get_logger
from matrix.shapelet_similarity import ShapeletSimilarityCalculator
from matrix.attention_evaluator import AttentionEvaluator
from classifier.shapelet_forest import ShapeletForest
from classifier.early_matcher import EarlyShapeletMatcher
from configs.config import FOREST_CONFIG

logger = get_logger(__name__)


class DistributionChangeDetector:
    """Detects changes in data distribution for incremental training triggers"""
    
    def __init__(self, window_size: int = 100, change_threshold: float = 0.1, min_samples: int = 50):
        """ Initialize distribution change detector
        
        Args:
            window_size: Size of sliding window
            change_threshold: Threshold for detecting changes
            min_samples: Minimum samples required for detection
        """
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.min_samples = min_samples
        
        # Data windows
        self.recent_window = deque(maxlen=window_size)
        self.historical_window = deque(maxlen=window_size)
        
        # Change statistics
        self.change_history = []
        self.last_change_time = 0
        
        logger.info(f"Initialized DistributionChangeDetector with window_size={window_size}")
    
    def add_sample(self, shapelet: np.ndarray, label: str, timestamp: float = None):
        """ Add sample to detector
        
        Args:
            shapelet: Input shapelet
            label: Predicted label
            timestamp: Sample timestamp
        """
        sample = {
            "shapelet": shapelet,
            "label": label,
            "timestamp": timestamp or time.time()
        }
        
        self.recent_window.append(sample)
        
        # Move to historical window when recent window is full
        if len(self.recent_window) >= self.window_size:
            old_sample = self.recent_window.popleft()
            self.historical_window.append(old_sample)
    
    def detect_change(self) -> Tuple[bool, float, str]:
        """ Detect distribution change
        
        Returns:
            Tuple of (change_detected, change_magnitude, change_type)
        """
        if len(self.recent_window) < self.min_samples or len(self.historical_window) < self.min_samples:
            return False, 0.0, "insufficient_data"
        
        # Calculate label distribution changes
        recent_labels = [s["label"] for s in self.recent_window]
        historical_labels = [s["label"] for s in self.historical_window]
        
        # Calculate distribution difference
        change_magnitude = self._calculate_distribution_difference(
            recent_labels, historical_labels
        )
        
        # Determine change type
        change_type = self._determine_change_type(recent_labels, historical_labels)
        
        # Detect change
        change_detected = change_magnitude > self.change_threshold
        
        if change_detected:
            self.change_history.append({
                "timestamp": time.time(),
                "magnitude": change_magnitude,
                "type": change_type
            })
            self.last_change_time = time.time()
            
            # Use debug level to avoid interfering with progress bars
            logger.debug(f"Distribution change detected: magnitude={change_magnitude:.3f}, type={change_type}")
        
        return change_detected, change_magnitude, change_type
    
    def _calculate_distribution_difference(self, recent_labels: List[str], historical_labels: List[str]) -> float:
        """Calculate difference between label distributions"""
        from collections import Counter
        
        recent_counts = Counter(recent_labels)
        historical_counts = Counter(historical_labels)
        
        # Get all unique labels
        all_labels = set(recent_counts.keys()) | set(historical_counts.keys())
        
        # Normalize to probabilities
        recent_total = len(recent_labels)
        historical_total = len(historical_labels)
        
        if recent_total == 0 or historical_total == 0:
            return 0.0
        
        # Calculate Jensen-Shannon divergence
        recent_probs = [recent_counts.get(label, 0) / recent_total for label in all_labels]
        historical_probs = [historical_counts.get(label, 0) / historical_total for label in all_labels]
        
        # Jensen-Shannon divergence
        js_divergence = self._jensen_shannon_divergence(recent_probs, historical_probs)
        
        return js_divergence
    
    def _jensen_shannon_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        def kl_divergence(p1: List[float], p2: List[float]) -> float:
            kl = 0.0
            for pi, pj in zip(p1, p2):
                if pi > 0 and pj > 0:
                    kl += pi * math.log2(pi / pj)
            return kl
        
        # Calculate mixture distribution
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        
        # Calculate JSD
        jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        
        return math.sqrt(jsd)  # Return square root for better interpretation
    
    def _determine_change_type(self, recent_labels: List[str], historical_labels: List[str]) -> str:
        """Determine type of distribution change"""
        from collections import Counter
        
        recent_counts = Counter(recent_labels)
        historical_counts = Counter(historical_labels)
        
        # Check for new classes
        new_classes = set(recent_counts.keys()) - set(historical_counts.keys())
        if new_classes:
            return f"new_classes:{','.join(new_classes)}"
        
        # Check for disappeared classes
        disappeared_classes = set(historical_counts.keys()) - set(recent_counts.keys())
        if disappeared_classes:
            return f"disappeared_classes:{','.join(disappeared_classes)}"
        
        # Check for proportion changes
        proportion_changes = []
        for label in set(recent_counts.keys()) & set(historical_counts.keys()):
            recent_prop = recent_counts[label] / len(recent_labels)
            historical_prop = historical_counts[label] / len(historical_labels)
            change = abs(recent_prop - historical_prop)
            
            if change > 0.1:  # Significant proportion change
                proportion_changes.append(f"{label}:{change:.2f}")
        
        if proportion_changes:
            return f"proportion_changes:{';'.join(proportion_changes)}"
        
        return "minor_shift"
    
    def get_change_history(self) -> List[Dict[str, any]]:
        """Get change detection history"""
        return self.change_history.copy()
    
    def reset_detector(self):
        """Reset detector state"""
        self.recent_window.clear()
        self.historical_window.clear()
        self.change_history.clear()
        self.last_change_time = 0
        
        logger.info("Reset distribution change detector")


class InformationGainCalculator:
    """Calculates information gain for shapelet selection"""
    
    def __init__(self, min_info_gain: float = 0.01):
        """ Initialize information gain calculator
        
        Args:
            min_info_gain: Minimum information gain threshold
        """
        self.min_info_gain = min_info_gain
        
        # Cache for efficiency
        self.entropy_cache = {}
        self.info_gain_cache = {}
        
        logger.info(f"Initialized InformationGainCalculator with min_info_gain={min_info_gain}")
    
    def calculate_information_gain(self, shapelet: np.ndarray, dataset: List[np.ndarray], 
                                  labels: List[str], threshold: float) -> float:
        """ Calculate information gain for shapelet split
        
        Args:
            shapelet: Candidate shapelet
            dataset: Dataset of shapelets
            labels: Corresponding labels
            threshold: Split threshold
            
        Returns:
            Information gain value
        """
        cache_key = (tuple(shapelet), tuple(map(tuple, dataset)), tuple(labels), threshold)
        
        if cache_key in self.info_gain_cache:
            return self.info_gain_cache[cache_key]
        
        # Calculate similarities
        similarities = [
            self._calculate_similarity(shapelet, data_shapelet) 
            for data_shapelet in dataset
        ]
        
        # Split dataset
        left_indices = [i for i, sim in enumerate(similarities) if sim < threshold]
        right_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
        
        if len(left_indices) < 2 or len(right_indices) < 2:
            return 0.0
        
        # Calculate information gain
        info_gain = self._calculate_information_gain(labels, left_indices, right_indices)
        
        # Cache result
        self.info_gain_cache[cache_key] = info_gain
        
        logger.debug(f"Information gain: {info_gain:.4f} for threshold {threshold:.3f}")
        return info_gain
    
    def _calculate_similarity(self, shapelet1: np.ndarray, shapelet2: np.ndarray) -> float:
        """Calculate similarity between two shapelets (simplified)"""
        from matrix.shapelet_similarity import ShapeletSimilarityCalculator
        
        calculator = ShapeletSimilarityCalculator()
        return calculator.calculate_similarity(shapelet1, shapelet2)
    
    def _calculate_information_gain(self, labels: List[str], left_indices: List[int], 
                                   right_indices: List[int]) -> float:
        """Calculate information gain for split"""
        # Calculate entropy before split
        total_entropy = self._calculate_entropy(labels)
        
        # Calculate weighted entropy after split
        left_labels = [labels[i] for i in left_indices]
        right_labels = [labels[i] for i in right_indices]
        
        left_entropy = self._calculate_entropy(left_labels)
        right_entropy = self._calculate_entropy(right_labels)
        
        left_weight = len(left_indices) / len(labels)
        right_weight = len(right_indices) / len(labels)
        
        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        
        # Information gain
        info_gain = total_entropy - weighted_entropy
        return info_gain
    
    def _calculate_entropy(self, labels: List[str]) -> float:
        """Calculate entropy of label distribution"""
        from collections import Counter
        
        if not labels:
            return 0.0
        
        cache_key = tuple(sorted(labels))
        if cache_key in self.entropy_cache:
            return self.entropy_cache[cache_key]
        
        label_counts = Counter(labels)
        total = len(labels)
        entropy = 0.0
        
        for count in label_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        self.entropy_cache[cache_key] = entropy
        return entropy
    
    def find_optimal_threshold(self, shapelet: np.ndarray, dataset: List[np.ndarray], 
                              labels: List[str]) -> Tuple[float, float]:
        """ Find optimal threshold for maximum information gain
        
        Args:
            shapelet: Candidate shapelet
            dataset: Dataset of shapelets
            labels: Corresponding labels
            
        Returns:
            Tuple of (optimal_threshold, max_information_gain)
        """
        # Calculate similarities
        similarities = [
            self._calculate_similarity(shapelet, data_shapelet) 
            for data_shapelet in dataset
        ]
        
        # Try different thresholds
        best_threshold = 0.5
        best_info_gain = 0.0
        
        for threshold in np.linspace(0.1, 0.9, 17):  # More granular search
            info_gain = self.calculate_information_gain(shapelet, dataset, labels, threshold)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.3f}, max info gain: {best_info_gain:.4f}")
        return best_threshold, best_info_gain
    
    def clear_cache(self):
        """Clear calculation cache"""
        self.entropy_cache.clear()
        self.info_gain_cache.clear()
        logger.debug("Cleared information gain cache")


class IncrementalTrainingClassifier:
    """Main incremental training classifier with forest updates"""
    
    def __init__(self, forest: ShapeletForest, early_matcher: EarlyShapeletMatcher, 
                 similarity_calculator: ShapeletSimilarityCalculator, 
                 attention_evaluator: AttentionEvaluator, 
                 change_detector: DistributionChangeDetector, 
                 info_gain_calculator: InformationGainCalculator, 
                 tree_score_threshold: float = 0.7, 
                 consistency_threshold: float = 0.6):
        """ Initialize incremental training classifier
        
        Args:
            forest: Shapelet forest
            early_matcher: Early shapelet matcher
            similarity_calculator: Similarity calculator
            attention_evaluator: Attention evaluator
            change_detector: Distribution change detector
            info_gain_calculator: Information gain calculator
            tree_score_threshold: Threshold for tree scoring
            consistency_threshold: Threshold for consistency checking
        """
        self.forest = forest
        self.early_matcher = early_matcher
        self.similarity_calculator = similarity_calculator
        self.attention_evaluator = attention_evaluator
        self.change_detector = change_detector
        self.info_gain_calculator = info_gain_calculator
        self.tree_score_threshold = tree_score_threshold
        self.consistency_threshold = consistency_threshold
        
        # Training statistics
        self.training_iterations = 0
        self.forest_updates = 0
        self.shapelet_additions = 0
        self.distribution_changes = 0
        
        # Candidate pools
        self.candidate_shapelets = []
        self.candidate_labels = []
        self.unlabeled_pool = []
        
        logger.info("Initialized IncrementalTrainingClassifier")
    
    def co_train(self, new_shapelets: List[np.ndarray], new_labels: List[str], 
                similarity_matrix: np.ndarray = None) -> Dict[str, any]:
        """ Perform incremental training iteration
        
        Args:
            new_shapelets: New training shapelets
            new_labels: Corresponding labels
            similarity_matrix: Updated similarity matrix
            
        Returns:
            Training statistics and results
        """
        logger.info(f"Starting incremental training iteration {self.training_iterations + 1}")
        
        # Add to candidate pools
        self.candidate_shapelets.extend(new_shapelets)
        self.candidate_labels.extend(new_labels)
        
        # Detect distribution changes
        for shapelet, label in zip(new_shapelets, new_labels):
            self.change_detector.add_sample(shapelet, label)
        
        change_detected, change_magnitude, change_type = self.change_detector.detect_change()
        
        if change_detected:
            self.distribution_changes += 1
            logger.info(f"Distribution change detected: {change_type} (magnitude: {change_magnitude:.3f})")
        
        # Evaluate forest performance
        forest_stats = self.forest.get_forest_statistics()
        current_avg_score = forest_stats.get("avg_tree_score", 0.0)
        
        # Check if forest update is needed
        update_needed = (
            change_detected or 
            current_avg_score < self.tree_score_threshold or 
            self.training_iterations % 10 == 0  # Periodic update
        )
        
        update_results = {}
        if update_needed:
            update_results = self._update_forest(similarity_matrix)
            self.forest_updates += 1
        
        # Select high-quality shapelets
        selected_shapelets, selected_labels = self._select_high_quality_shapelets(
            new_shapelets, new_labels
        )
        
        # Update early matcher cache
        for shapelet, label in zip(selected_shapelets, selected_labels):
            self.early_matcher.add_cached_shapelet(shapelet, label)
        
        self.shapelet_additions += len(selected_shapelets)
        self.training_iterations += 1
        
        # Compile results
        results = {
            "iteration": self.training_iterations,
            "distribution_change_detected": change_detected,
            "change_magnitude": change_magnitude,
            "change_type": change_type,
            "forest_update_performed": update_needed,
            "update_results": update_results,
            "selected_shapelets": len(selected_shapelets),
            "total_shapelets": len(self.candidate_shapelets),
            "forest_avg_score": current_avg_score,
            "statistics": self.get_training_statistics()
        }
        
        logger.info(f"Incremental training iteration completed: {results}")
        return results
    
    def _update_forest(self, similarity_matrix: np.ndarray = None) -> Dict[str, any]:
        """ Update forest based on incremental training principles
        
        Args:
            similarity_matrix: Updated similarity matrix
            
        Returns:
            Update results
        """
        logger.info("Performing forest update")
        
        # Get current tree scores
        current_scores = {
            tree.tree_id: tree.get_tree_score() 
            for tree in self.forest.trees
        }
        
        # Identify trees to update
        trees_to_update = [
            (tree_id, score) 
            for tree_id, score in current_scores.items() 
            if score < self.tree_score_threshold
        ]
        
        logger.info(f"Updating {len(trees_to_update)} low-performing trees")
        
        # Select new shapelets for retraining
        new_training_data = self._select_training_shapelets()
        
        # Update forest
        self.forest.update_forest(
            new_training_data["shapelets"], 
            new_training_data["labels"], 
            similarity_matrix
        )
        
        # Get updated scores
        updated_scores = {
            tree.tree_id: tree.get_tree_score() 
            for tree in self.forest.trees
        }
        
        # Calculate improvement
        improvements = {}
        for tree_id in current_scores.keys():
            if tree_id in updated_scores:
                improvement = updated_scores[tree_id] - current_scores[tree_id]
                improvements[tree_id] = improvement
        
        avg_improvement = np.mean(list(improvements.values())) if improvements else 0.0
        
        update_results = {
            "trees_updated": len(trees_to_update),
            "new_training_samples": len(new_training_data["shapelets"]),
            "avg_score_improvement": avg_improvement,
            "score_improvements": improvements,
            "updated_forest_stats": self.forest.get_forest_statistics()
        }
        
        logger.info(f"Forest update completed: avg improvement = {avg_improvement:.3f}")
        return update_results
    
    def _select_high_quality_shapelets(self, shapelets: List[np.ndarray], labels: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """ Select high-quality shapelets based on information gain
        
        Args:
            shapelets: Candidate shapelets
            labels: Corresponding labels
            
        Returns:
            Tuple of (selected_shapelets, selected_labels)
        """
        if len(shapelets) < 3:
            return shapelets, labels
        
        selected_shapelets = []
        selected_labels = []
        
        for i, (shapelet, label) in enumerate(zip(shapelets, labels)):
            # Calculate information gain
            optimal_threshold, info_gain = self.info_gain_calculator.find_optimal_threshold(
                shapelet, shapelets, labels
            )
            
            # Select if information gain is significant
            if info_gain >= self.info_gain_calculator.min_info_gain:
                selected_shapelets.append(shapelet)
                selected_labels.append(label)
                
                logger.debug(f"Selected shapelet {i}: info_gain={info_gain:.4f}, threshold={optimal_threshold:.3f}")
        
        logger.info(f"Selected {len(selected_shapelets)} high-quality shapelets from {len(shapelets)} candidates")
        return selected_shapelets, selected_labels
    
    def _select_training_shapelets(self) -> Dict[str, List]:
        """ Select training shapelets for forest update
        
        Returns:
            Dictionary with "shapelets" and "labels" keys
        """
        if not self.candidate_shapelets:
            return {"shapelets": [], "labels": []}
        
        # Use attention evaluation to select most relevant shapelets
        if len(self.candidate_shapelets) > 10:
            # Select top shapelets based on attention weights
            attention_weights = self.attention_evaluator.get_top_attention_labels(k=5)
            
            # Filter shapelets by high-attention labels
            selected_indices = []
            for i, label in enumerate(self.candidate_labels):
                if any(label == att_label for att_label, _ in attention_weights):
                    selected_indices.append(i)
            
            # If not enough high-attention shapelets, add random selection
            if len(selected_indices) < 10:
                remaining_indices = [i for i in range(len(self.candidate_shapelets)) if i not in selected_indices]
                additional_needed = min(10 - len(selected_indices), len(remaining_indices))
                selected_indices.extend(np.random.choice(remaining_indices, additional_needed, replace=False))
        else:
            # Use all available shapelets
            selected_indices = list(range(len(self.candidate_shapelets)))
        
        selected_shapelets = [self.candidate_shapelets[i] for i in selected_indices]
        selected_labels = [self.candidate_labels[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_shapelets)} training shapelets")
        return {"shapelets": selected_shapelets, "labels": selected_labels}
    
    def classify_with_incremental_training(self, input_shapelet: np.ndarray, 
                                         use_early_match: bool = True) -> Dict[str, any]:
        """ Classify with incremental training enhancements
        
        Args:
            input_shapelet: Input shapelet to classify
            use_early_match: Whether to use early matching
            
        Returns:
            Classification results with incremental training information
        """
        # Try early matching first
        early_result = None
        if use_early_match:
            early_result = self.early_matcher.try_early_match(input_shapelet)
        
        if early_result:
            predicted_label, confidence, matched_shapelet = early_result
            
            logger.info(f"Early match successful: {predicted_label} (confidence: {confidence:.3f})")
            
            return {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "method": "early_match",
                "matched_shapelet": matched_shapelet,
                "incremental_training_info": {
                    "distribution_change_detected": False,
                    "forest_update_needed": False
                }
            }
        
        # Use forest classification
        predicted_label, confidence, vote_distribution = self.forest.classify(input_shapelet)
        
        # Add to change detector
        if predicted_label:
            self.change_detector.add_sample(input_shapelet, predicted_label)
        
        # Check for distribution changes
        change_detected, change_magnitude, change_type = self.change_detector.detect_change()
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "method": "forest_classification",
            "vote_distribution": vote_distribution,
            "incremental_training_info": {
                "distribution_change_detected": change_detected,
                "change_magnitude": change_magnitude,
                "change_type": change_type,
                "forest_update_needed": change_detected or self.training_iterations % 10 == 0
            }
        }
    
    def classify_with_incremental_training_advanced(self, input_shapelet: np.ndarray, 
                                                  use_early_match: bool = True, 
                                                  calculate_earliness: bool = False, 
                                                  full_series: np.ndarray = None) -> Dict[str, any]:
        """ Classify with incremental training and optional Earliness calculation
        
        Args:
            input_shapelet: Input shapelet to classify
            use_early_match: Whether to use early matching
            calculate_earliness: Whether to calculate Earliness metric
            full_series: Complete time series for Earliness calculation (required if calculate_earliness=True)
            
        Returns:
            Classification results with incremental training information and optional Earliness
        """
        # Basic classification with incremental training - call the original method
        result = self.classify_with_incremental_training(input_shapelet, use_early_match)
        
        # Add Earliness calculation if requested and full series is provided
        if calculate_earliness and full_series is not None and hasattr(self, 'early_classifier'):
            try:
                # Use the early classifier to get progressive classification with Earliness
                early_result = self.early_classifier.classify_with_progressive_observation(
                    full_series, self.early_matcher, self.forest
                )
                
                # Merge Earliness information into the result
                result.update({
                    "earliness": early_result.get("earliness", 1.0),
                    "observation_ratio": early_result.get("observation_ratio", 1.0),
                    "classification_method": early_result.get("method", "complete_classification"),
                    "used_early_match": early_result.get("used_early_match", False),
                    "classification_point": early_result.get("classification_point", len(full_series))
                })
                
                logger.info(f"Earliness calculated: {result['earliness']:.3f} "
                           f"(observation ratio: {result['observation_ratio']:.1%})")
                
            except Exception as e:
                logger.warning(f"Failed to calculate Earliness: {e}")
                # Fallback to default values
                result.update({
                    "earliness": 1.0,
                    "observation_ratio": 1.0,
                    "classification_method": "complete_classification",
                    "used_early_match": False,
                    "classification_point": len(full_series) if full_series is not None else 0
                })
        else:
            # Add default Earliness values when not calculating
            result.update({
                "earliness": 1.0,
                "observation_ratio": 1.0,
                "classification_method": result.get("method", "forest_classification"),
                "used_early_match": result.get("method") == "early_match",
                "classification_point": len(full_series) if full_series is not None else 0
            })
        
        return result
    
    def get_training_statistics(self) -> Dict[str, any]:
        """Get training statistics"""
        return {
            "training_iterations": self.training_iterations,
            "forest_updates": self.forest_updates,
            "shapelet_additions": self.shapelet_additions,
            "distribution_changes": self.distribution_changes,
            "candidate_pool_size": len(self.candidate_shapelets),
            "early_matcher_stats": self.early_matcher.get_cache_statistics(),
            "change_history": self.change_detector.get_change_history()[-10:]  # Last 10 changes
        }
    
    def reset_incremental_training(self):
        """Reset incremental training state"""
        self.training_iterations = 0
        self.forest_updates = 0
        self.shapelet_additions = 0
        self.distribution_changes = 0
        
        self.candidate_shapelets.clear()
        self.candidate_labels.clear()
        self.unlabeled_pool.clear()
        
        self.change_detector.reset_detector()
        self.info_gain_calculator.clear_cache()
        
        logger.info("Reset incremental training classifier")


# Utility functions
def create_incremental_training_classifier(forest: ShapeletForest, 
                                         early_matcher: EarlyShapeletMatcher, 
                                         similarity_calculator: ShapeletSimilarityCalculator, 
                                         attention_evaluator: AttentionEvaluator, 
                                         config: Dict[str, any] = None) -> IncrementalTrainingClassifier:
    """Factory function to create incremental training classifier"""
    
    if config is None:
        config = FOREST_CONFIG
    
    change_detector = DistributionChangeDetector(
        window_size=config.get("change_window_size", 100),
        change_threshold=config.get("change_threshold", 0.1),
        min_samples=config.get("min_change_samples", 50)
    )
    
    info_gain_calculator = InformationGainCalculator(
        min_info_gain=config.get("info_gain_threshold", 0.01)
    )
    
    return IncrementalTrainingClassifier(
        forest=forest,
        early_matcher=early_matcher,
        similarity_calculator=similarity_calculator,
        attention_evaluator=attention_evaluator,
        change_detector=change_detector,
        info_gain_calculator=info_gain_calculator,
        tree_score_threshold=config.get("tree_score_threshold", 0.7),
        consistency_threshold=config.get("consistency_threshold", 0.6),
    )