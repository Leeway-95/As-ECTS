import numpy as np
from typing import Dict, Optional, Any
from utils.logger import get_logger
from classifier.early_matcher import EarlyShapeletMatcher

logger = get_logger(__name__)


class As_ECTS:
    """As-ECTS implementation for early classification of time series"""
    
    def __init__(self, early_threshold: float = 0.85, min_observation_ratio: float = 0.3, 
                 confidence_threshold: float = 0.7, step_size: int = 1):
        """ Initialize As-ECTS
        
        Args:
            early_threshold: Similarity threshold for early decision
            min_observation_ratio: Minimum observation ratio before allowing early decision
            confidence_threshold: Minimum confidence for early classification
            step_size: Number of samples to add in each progressive step
        """
        self.early_threshold = early_threshold
        self.min_observation_ratio = min_observation_ratio
        self.confidence_threshold = confidence_threshold
        self.step_size = step_size
        
        # Statistics
        self.total_classifications = 0
        self.early_classifications = 0
        self.avg_earliness = 0.0
        
        logger.info(f"Initialized As-ECTS with threshold={early_threshold}")

    def classify_with_progressive_observation(self, full_series: np.ndarray, 
                                            early_matcher: EarlyShapeletMatcher, 
                                            forest_classifier) -> Dict[str, Any]:
        """ Classify time series with progressive observation and Earliness calculation
        
        Args:
            full_series: Complete time series data
            early_matcher: Early matcher for cached shapelets
            forest_classifier: Forest classifier for complete classification
            
        Returns:
            Classification results with Earliness metric
        """
        self.total_classifications += 1
        series_length = len(full_series)
        min_length = int(series_length * self.min_observation_ratio)
        
        # Progressive observation loop with safety limits
        max_iterations = min(100, series_length)  # Prevent infinite loops
        iteration_count = 0
        
        for current_length in range(min_length, series_length + 1, self.step_size):
            iteration_count += 1
            if iteration_count > max_iterations:
                logger.warning(f"Reached maximum iterations ({max_iterations}), forcing complete classification")
                break
            
            # Get current partial observation
            partial_series = full_series[:current_length]
            observation_ratio = current_length / series_length
            
            # Try early matching with partial data
            early_result = self._try_early_match_with_partial_data(
                partial_series, observation_ratio, early_matcher
            )
            
            if early_result:
                self.early_classifications += 1

                base_earliness = 1.0 - observation_ratio
                scaled_earliness = 0.1 + (base_earliness * 0.8)
                earliness = max(0.1, min(0.9, scaled_earliness))
                
                logger.info(f" Early classification at {observation_ratio:.1%} observation "
                           f"(Earliness: {earliness:.3f}, Classification point: {current_length}/{series_length})")
                
                return {
                    "predicted_label": early_result["label"],
                    "confidence": early_result["confidence"],
                    "method": "early_classification",
                    "earliness": earliness,
                    "observation_ratio": observation_ratio,
                    "used_early_match": True,
                    "classification_point": current_length
                }
            
            # Check if we have enough confidence for early decision
            if observation_ratio >= 0.8:  # At 80% observation, force a decision
                logger.debug(f"Reached 80% observation ratio, forcing decision")
                break
        
        # Fall back to complete classification
        final_result = forest_classifier.classify(full_series)
        earliness = 0.0  # No early classification achieved
        
        return {
            "predicted_label": final_result[0],
            "confidence": final_result[1],
            "method": "complete_classification",
            "earliness": earliness,
            "observation_ratio": 1.0,
            "used_early_match": False,
            "classification_point": series_length
        }

    def _try_early_match_with_partial_data(self, partial_series: np.ndarray, 
                                          observation_ratio: float, 
                                          early_matcher: EarlyShapeletMatcher) -> Optional[Dict[str, Any]]:
        """ Try to match with partial data for early classification
        
        Args:
            partial_series: Partial time series observation
            observation_ratio: Current observation ratio
            early_matcher: Early matcher instance
            
        Returns:
            Early match result or None
        """
        # Only try early matching if we have sufficient observation
        if observation_ratio < self.min_observation_ratio:
            return None
        
        # Create a synthetic shapelet from partial data
        partial_shapelet = self._create_partial_shapelet(partial_series)
        
        # Try early matching
        early_match = early_matcher.try_early_match(partial_shapelet)
        
        if early_match:
            predicted_label, confidence, matched_shapelet = early_match
            
            # Check if confidence is sufficient for early decision
            if confidence >= self.confidence_threshold:
                return {
                    "label": predicted_label,
                    "confidence": confidence,
                    "matched_shapelet": matched_shapelet,
                    "observation_ratio": observation_ratio
                }
        
        return None

    def _create_partial_shapelet(self, partial_series: np.ndarray) -> np.ndarray:
        """ Create a shapelet from partial time series data
        
        Args:
            partial_series: Partial time series
            
        Returns:
            Shapelet representation
        """
        # Simple approach: normalize and return as shapelet
        # In practice, this could be more sophisticated
        if len(partial_series) == 0:
            return np.array([])
        
        # Normalize to standard scale
        normalized = (partial_series - np.mean(partial_series)) / (np.std(partial_series) + 1e-8)
        return normalized

class StreamingEarlyClassifier(As_ECTS):
    """Streaming version of progressive early classifier"""
    
    def __init__(self, early_threshold: float = 0.85, min_observation_ratio: float = 0.3, 
                 confidence_threshold: float = 0.7, step_size: int = 1, 
                 window_size: int = 100, decay_factor: float = 0.95):
        """ Initialize streaming early classifier
        
        Args:
            early_threshold: Similarity threshold for early decision
            min_observation_ratio: Minimum observation ratio
            confidence_threshold: Minimum confidence threshold
            step_size: Progressive step size
            window_size: Sliding window size for streaming
            decay_factor: Decay factor for adaptive thresholds
        """
        super().__init__(early_threshold, min_observation_ratio, confidence_threshold, step_size)
        
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # Streaming statistics
        self.recent_earliness_scores = []
        self.adaptive_threshold = early_threshold
        
        logger.info(f"Initialized As-ECTS with window_size={window_size}")

    def classify_streaming_data(self, streaming_data: np.ndarray, 
                               early_matcher: EarlyShapeletMatcher, 
                               forest_classifier) -> Dict[str, Any]:
        """ Classify streaming data with adaptive early classification
        
        Args:
            streaming_data: Streaming time series data
            early_matcher: Early matcher
            forest_classifier: Forest classifier
            
        Returns:
            Classification results with adaptive earliness
        """
        # Update adaptive threshold based on recent performance
        self._update_adaptive_threshold()
        
        # Use parent's progressive classification
        result = self.classify_with_progressive_observation(
            streaming_data, early_matcher, forest_classifier
        )
        
        # Update streaming statistics
        if "earliness" in result:
            self.recent_earliness_scores.append(result["earliness"])
            if len(self.recent_earliness_scores) > self.window_size:
                self.recent_earliness_scores.pop(0)
            
            # Update average earliness
            if self.recent_earliness_scores:
                self.avg_earliness = np.mean(self.recent_earliness_scores)
        
        return result

    def _update_adaptive_threshold(self):
        """Update adaptive threshold based on recent performance"""
        if len(self.recent_earliness_scores) < 10:
            return
        
        recent_avg = np.mean(self.recent_earliness_scores[-10:])
        
        # Adjust threshold based on recent performance
        if recent_avg > 0.7:  # High earliness (good early classification)
            self.adaptive_threshold = min(self.adaptive_threshold * 1.05, 0.95)
        elif recent_avg < 0.3:  # Low earliness (late classification)
            self.adaptive_threshold = max(self.adaptive_threshold * 0.95, 0.5)
        
        logger.debug(f"Updated adaptive threshold to {self.adaptive_threshold:.3f}")