import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
from utils.logger import get_logger
from matrix.shapelet_similarity import ShapeletSimilarityCalculator
from configs.config import EARLY_CONFIG
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import warnings

logger = get_logger(__name__)


class CachedShapelet:
    def __init__(self, shapelet: np.ndarray, label: str, confidence: float = 1.0, 
                 timestamp: float = None, usage_count: int = 0, 
                 discriminative_score: float = 0.0, length: int = None):
        """ Initialize cached shapelet
        
        Args:
            shapelet: The shapelet data
            label: Predicted label
            confidence: Prediction confidence
            timestamp: Creation timestamp
            usage_count: Number of times used
            discriminative_score: Discriminative power score
            length: Shapelet length
        """
        self.shapelet = np.array(shapelet)
        self.label = label
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.usage_count = usage_count
        self.last_accessed = self.timestamp
        self.similarity_scores = {}  # Cache similarity scores
        self.discriminative_score = discriminative_score
        self.length = length or len(shapelet)
        self.match_success_rate = 0.0  # Track matching success
        self.total_matches = 0
        self.successful_matches = 0

    def update_access(self, success: bool = True):
        """Update access statistics"""
        self.usage_count += 1
        self.last_accessed = time.time()
        self.total_matches += 1
        if success:
            self.successful_matches += 1
        self.match_success_rate = self.successful_matches / max(self.total_matches, 1)

    def update_confidence(self, new_confidence: float, learning_rate: float = 0.1):
        """Update confidence with learning rate"""
        self.confidence = (1 - learning_rate) * self.confidence + learning_rate * new_confidence

    def add_similarity_score(self, other_shapelet: np.ndarray, similarity: float):
        """Cache similarity score with another shapelet"""
        key = tuple(other_shapelet)
        self.similarity_scores[key] = similarity

    def get_similarity_score(self, other_shapelet: np.ndarray) -> Optional[float]:
        """Get cached similarity score"""
        key = tuple(other_shapelet)
        return self.similarity_scores.get(key)


class AdvancedSimilarityCalculator:
    """Advanced similarity calculator with multiple metrics"""
    
    def __init__(self, distance_threshold: float = 0.1, similarity_threshold: float = 0.7):
        """ Initialize advanced similarity calculator
        
        Args:
            distance_threshold: Threshold for distance-based metrics
            similarity_threshold: Threshold for similarity decisions
        """
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.base_calculator = ShapeletSimilarityCalculator(distance_threshold, similarity_threshold)

    def calculate_multi_metric_similarity(self, shapelet1: np.ndarray, shapelet2: np.ndarray) -> float:
        """ Calculate similarity using multiple metrics
        
        Args:
            shapelet1: First shapelet
            shapelet2: Second shapelet
            
        Returns:
            Combined similarity score (0-1)
        """
        try:
            # 1. Distance-based similarity
            distance_sim = self.base_calculator.calculate_similarity(shapelet1, shapelet2)
            
            # 2. Cosine similarity (shape pattern) - handle different lengths
            min_len = min(len(shapelet1), len(shapelet2))
            if min_len > 1:
                shapelet1_trunc = shapelet1[:min_len].reshape(1, -1)
                shapelet2_trunc = shapelet2[:min_len].reshape(1, -1)
                cosine_sim = cosine_similarity(shapelet1_trunc, shapelet2_trunc)[0, 0]
            else:
                cosine_sim = 0.0
            
            # 3. Pearson correlation (temporal correlation)
            min_len = min(len(shapelet1), len(shapelet2))
            if min_len > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr_coeff, _ = pearsonr(shapelet1[:min_len], shapelet2[:min_len])
                    pearson_sim = max(0, (corr_coeff + 1) / 2)  # Normalize to 0-1
            else:
                pearson_sim = 0.0
            
            # 4. Length similarity (penalize large length differences)
            length_ratio = min(len(shapelet1), len(shapelet2)) / max(len(shapelet1), len(shapelet2))
            length_sim = length_ratio ** 2  # Quadratic penalty for large differences
            
            # 5. Dynamic weighting based on shapelet characteristics
            weights = self._calculate_adaptive_weights(shapelet1, shapelet2)
            
            # Combined similarity with adaptive weighting
            combined_sim = (
                weights['distance'] * distance_sim +
                weights['cosine'] * cosine_sim +
                weights['pearson'] * pearson_sim +
                weights['length'] * length_sim
            )
            
            # Normalize to 0-1 range
            final_sim = max(0.0, min(1.0, combined_sim))
            
            logger.debug(f"Multi-metric similarity: distance={distance_sim:.3f}, "
                        f"cosine={cosine_sim:.3f}, pearson={pearson_sim:.3f}, "
                        f"length={length_sim:.3f}, final={final_sim:.3f}")
            return final_sim
            
        except Exception as e:
            logger.error(f"Error calculating multi-metric similarity: {e}")
            return 0.0

    def _calculate_adaptive_weights(self, shapelet1: np.ndarray, shapelet2: np.ndarray) -> Dict[str, float]:
        """Calculate adaptive weights based on shapelet characteristics"""
        # Base weights
        weights = {
            'distance': 0.3,
            'cosine': 0.3,
            'pearson': 0.2,
            'length': 0.2
        }
        
        # Adjust weights based on characteristics
        len1, len2 = len(shapelet1), len(shapelet2)
        
        # If lengths are very different, reduce length weight and increase pattern weights
        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < 0.5:
            weights['length'] *= 0.5
            weights['cosine'] *= 1.2
            weights['pearson'] *= 1.1
        # If lengths are similar, increase length and distance weights
        elif length_ratio > 0.8:
            weights['length'] *= 1.2
            weights['distance'] *= 1.1
            
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        return weights


class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on performance feedback"""
    
    def __init__(self, initial_threshold: float = 0.6, min_threshold: float = 0.4, 
                 max_threshold: float = 0.9, adjustment_rate: float = 0.02):
        """ Initialize adaptive threshold manager
        
        Args:
            initial_threshold: Initial similarity threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            adjustment_rate: Rate of threshold adjustment
        """
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjustment_rate = adjustment_rate
        
        # Performance tracking
        self.recent_matches = deque(maxlen=50)
        self.accuracy_history = deque(maxlen=100)
        self.earliness_history = deque(maxlen=100)

    def update_threshold(self, match_success: bool, prediction_correct: bool, 
                        earliness: float, confidence: float):
        """Update threshold based on match performance"""
        self.recent_matches.append({
            'success': match_success,
            'correct': prediction_correct,
            'earliness': earliness,
            'confidence': confidence
        })
        
        if len(self.recent_matches) >= 10:  # Need minimum data
            # Calculate performance metrics
            recent_accuracy = sum(1 for m in self.recent_matches if m['correct']) / len(self.recent_matches)
            recent_earliness = sum(m['earliness'] for m in self.recent_matches) / len(self.recent_matches)
            
            self.accuracy_history.append(recent_accuracy)
            self.earliness_history.append(recent_earliness)
            
            # Adjust threshold based on performance
            if recent_accuracy < 0.6:  # Low accuracy - lower threshold
                self.current_threshold = max(self.min_threshold, self.current_threshold - self.adjustment_rate)
                logger.info(f"Lowering threshold to {self.current_threshold:.3f} due to low accuracy ({recent_accuracy:.3f})")
            elif recent_accuracy > 0.85 and recent_earliness < 0.3:  # High accuracy and good earliness - increase threshold
                self.current_threshold = min(self.max_threshold, self.current_threshold + self.adjustment_rate)
                logger.info(f"Raising threshold to {self.current_threshold:.3f} due to high performance")

    def get_current_threshold(self) -> float:
        """Get current adaptive threshold"""
        return self.current_threshold


class EarlyShapeletMatcher:
    """Early shapelet matcher with improved algorithms"""
    
    def __init__(self, early_match_threshold: float = 0.6, max_lookback: int = 10, 
                 min_confidence: float = 0.5, cache_size: int = 200, 
                 progress_manager=None, enable_adaptive_threshold: bool = True):
        """ Initialize early shapelet matcher
        
        Args:
            early_match_threshold: Initial similarity threshold for early matching
            max_lookback: Maximum number of previous shapelets to consider
            min_confidence: Minimum confidence for early decision
            cache_size: Maximum size of shapelet cache
            progress_manager: Optional progress manager
            enable_adaptive_threshold: Enable adaptive threshold adjustment
        """
        self.initial_threshold = early_match_threshold
        self.max_lookback = max_lookback
        self.min_confidence = min_confidence
        self.cache_size = cache_size
        self.progress_manager = progress_manager
        self.enable_adaptive_threshold = enable_adaptive_threshold
        
        # Caching structures
        self.shapelet_cache: Deque[CachedShapelet] = deque(maxlen=cache_size)
        self.label_cache: Dict[str, List[CachedShapelet]] = {}
        self.high_quality_cache: List[CachedShapelet] = []  # Top performers
        
        # Advanced components
        self.similarity_calculator = AdvancedSimilarityCalculator(
            similarity_threshold=early_match_threshold
        )
        self.threshold_manager = AdaptiveThresholdManager(
            initial_threshold=early_match_threshold,
            min_threshold=0.4,
            max_threshold=0.85,
            adjustment_rate=0.01
        )
        
        # Statistics
        self.total_matches = 0
        self.early_matches = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Progress tracking
        self._cache_task_id = None
        self._last_cache_size = 0
        
        logger.info(f"Initialized EarlyShapeletMatcher with adaptive_threshold={enable_adaptive_threshold}")

    def try_early_match(self, input_shapelet: np.ndarray, true_label: Optional[str] = None, 
                       current_label: Optional[str] = None) -> Optional[Tuple[str, float, CachedShapelet]]:
        """ Early matching with adaptive threshold and intelligent candidate selection
        
        Args:
            input_shapelet: Input shapelet to classify
            true_label: True label for performance feedback (optional)
            current_label: Optional current label for context
            
        Returns:
            Tuple of (predicted_label, confidence, matched_cached_shapelet) or None
        """
        self.total_matches += 1
        
        if not self.shapelet_cache:
            logger.debug("No cached shapelets available")
            self.cache_misses += 1
            return None
            
        # Get adaptive threshold
        current_threshold = self.threshold_manager.get_current_threshold() if self.enable_adaptive_threshold else self.initial_threshold
        
        # Get candidate shapelets
        candidates = self._get_candidates(input_shapelet, current_label)
        if not candidates:
            logger.debug("No suitable candidates for early matching")
            self.cache_misses += 1
            return None
            
        # Find best match using multi-metric similarity
        best_match = self._find_best_match(input_shapelet, candidates)
        if best_match is None:
            self.cache_misses += 1
            return None
            
        matched_shapelet, similarity, confidence = best_match
        
        # Check if similarity exceeds adaptive threshold
        if similarity >= current_threshold and confidence >= self.min_confidence:
            self.early_matches += 1
            self.cache_hits += 1
            
            # Update access statistics
            prediction_correct = true_label is not None and matched_shapelet.label == true_label
            matched_shapelet.update_access(success=prediction_correct)
            
            # Update performance metrics
            if true_label is not None:
                self.total_predictions += 1
                if prediction_correct:
                    self.correct_predictions += 1
            
            # Update adaptive threshold
            if self.enable_adaptive_threshold and true_label is not None:
                earliness = 0.5  # Placeholder - should be calculated based on position in sequence
                self.threshold_manager.update_threshold(
                    match_success=True,
                    prediction_correct=prediction_correct,
                    earliness=earliness,
                    confidence=confidence
                )
            
            logger.info(f"Early match: label={matched_shapelet.label}, "
                       f"similarity={similarity:.3f}, threshold={current_threshold:.3f}, "
                       f"confidence={confidence:.3f}")
            return matched_shapelet.label, confidence, matched_shapelet
            
        self.cache_misses += 1
        logger.debug(f"No early match: best similarity={similarity:.3f}, "
                    f"threshold={current_threshold:.3f}, confidence={confidence:.3f}")
        return None

    def _get_candidates(self, input_shapelet: np.ndarray, current_label: Optional[str]) -> List[CachedShapelet]:
        """ Candidate selection with intelligent filtering and ranking
        
        Args:
            input_shapelet: Input shapelet
            current_label: Optional label for filtering
            
        Returns:
            List of candidate cached shapelets
        """
        candidates = []
        input_length = len(input_shapelet)
        
        # 1. Length-based filtering (within reasonable range)
        length_tolerance = 0.3  # 30% length difference tolerance
        min_length = int(input_length * (1 - length_tolerance))
        max_length = int(input_length * (1 + length_tolerance))
        
        # 2. Multi-stage candidate selection
        candidate_scores = []
        
        # Stage 1: Get potential candidates from different sources
        potential_candidates = []
        
        # From same-label cache (high priority)
        if current_label and current_label in self.label_cache:
            label_candidates = self.label_cache[current_label]
            for candidate in label_candidates:
                if min_length <= candidate.length <= max_length:
                    potential_candidates.append(candidate)
        
        # From high-quality cache (medium priority)
        for candidate in self.high_quality_cache:
            if min_length <= candidate.length <= max_length:
                potential_candidates.append(candidate)
        
        # From recent cache (lower priority)
        recent_candidates = list(self.shapelet_cache)[-self.max_lookback*2:]
        for candidate in recent_candidates:
            if min_length <= candidate.length <= max_length:
                potential_candidates.append(candidate)
        
        # Stage 2: Score and rank candidates
        for candidate in potential_candidates:
            score = self._calculate_candidate_score(candidate, input_shapelet, current_label)
            candidate_scores.append((candidate, score))
        
        # Stage 3: Select top candidates
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates, ensuring diversity
        selected_candidates = []
        seen_labels = set()
        for candidate, score in candidate_scores[:self.max_lookback * 2]:
            if score > 0.1:  # Minimum quality threshold
                # Ensure label diversity
                if candidate.label not in seen_labels or len(selected_candidates) < self.max_lookback:
                    selected_candidates.append(candidate)
                    seen_labels.add(candidate.label)
                if len(selected_candidates) >= self.max_lookback * 1.5:
                    break
        
        return selected_candidates[:self.max_lookback]

    def _calculate_candidate_score(self, candidate: CachedShapelet, input_shapelet: np.ndarray, 
                                  current_label: Optional[str]) -> float:
        """ Calculate comprehensive candidate score
        
        Args:
            candidate: Candidate cached shapelet
            input_shapelet: Input shapelet
            current_label: Optional current label
            
        Returns:
            Candidate quality score
        """
        score = 0.0
        
        # 1. Success rate score (30%)
        success_rate = candidate.match_success_rate
        score += success_rate * 0.3
        
        # 2. Confidence score (20%)
        score += candidate.confidence * 0.2
        
        # 3. Discriminative score (20%)
        score += candidate.discriminative_score * 0.2
        
        # 4. Recency score (15%)
        time_decay = 1.0 / (1.0 + (time.time() - candidate.last_accessed) / 3600)  # 1-hour half-life
        score += time_decay * 0.15
        
        # 5. Label match bonus (15%)
        if current_label and candidate.label == current_label:
            score += 0.15
            
        return score

    def _find_best_match(self, input_shapelet: np.ndarray, candidates: List[CachedShapelet]) -> Optional[Tuple[CachedShapelet, float, float]]:
        """ Find best matching shapelet using multi-metric similarity
        
        Args:
            input_shapelet: Input shapelet to match
            candidates: List of candidate cached shapelets
            
        Returns:
            Tuple of (best_match, similarity, confidence) or None
        """
        best_match = None
        best_similarity = -1.0
        best_confidence = 0.0
        
        for candidate in candidates:
            # Check cached similarity first
            cached_similarity = candidate.get_similarity_score(input_shapelet)
            if cached_similarity is not None:
                similarity = cached_similarity
                logger.debug(f"Using cached similarity: {similarity:.3f}")
            else:
                # Calculate multi-metric similarity
                similarity = self.similarity_calculator.calculate_multi_metric_similarity(
                    input_shapelet, candidate.shapelet
                )
                # Cache for future use
                candidate.add_similarity_score(input_shapelet, similarity)
            
            # Calculate confidence based on similarity and candidate quality
            confidence = self._calculate_match_confidence(similarity, candidate)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate
                best_confidence = confidence
        
        if best_match:
            logger.debug(f"Best match: similarity={best_similarity:.3f}, "
                        f"confidence={best_confidence:.3f}, label={best_match.label}")
            return (best_match, best_similarity, best_confidence)
        
        return None

    def _calculate_match_confidence(self, similarity: float, candidate: CachedShapelet) -> float:
        """ Calculate match confidence based on similarity and candidate characteristics
        
        Args:
            similarity: Raw similarity score
            candidate: Matched candidate
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from similarity
        base_confidence = similarity
        
        # Adjust based on candidate quality
        quality_factor = (
            candidate.confidence * 0.4 +
            candidate.match_success_rate * 0.3 +
            candidate.discriminative_score * 0.3
        )
        
        # Combine factors
        final_confidence = base_confidence * (0.7 + 0.3 * quality_factor)
        return max(0.0, min(1.0, final_confidence))

    def add_cached_shapelet(self, shapelet: np.ndarray, label: str, confidence: float = 1.0, 
                           timestamp: float = None, discriminative_score: float = 0.0):
        """ Shapelet caching with quality assessment
        
        Args:
            shapelet: Shapelet to cache
            label: Predicted label
            confidence: Prediction confidence
            timestamp: Creation timestamp
            discriminative_score: Discriminative power score
        """
        cached_shapelet = CachedShapelet(
            shapelet, label, confidence, timestamp, 
            discriminative_score=discriminative_score
        )
        
        # Add to general cache
        self.shapelet_cache.append(cached_shapelet)
        
        # Add to label-specific cache
        if label not in self.label_cache:
            self.label_cache[label] = []
        self.label_cache[label].append(cached_shapelet)
        
        # Update high-quality cache
        self._update_high_quality_cache(cached_shapelet)
        
        # Maintain cache size limits
        self._maintain_cache_size()
        
        # Update progress if available
        self._update_progress()
        
        logger.debug(f"Added cached shapelet for label {label}")

    def _update_high_quality_cache(self, shapelet: CachedShapelet):
        """Update high-quality cache with top performers"""
        self.high_quality_cache.append(shapelet)
        
        # Keep only top performers based on success rate and confidence
        self.high_quality_cache.sort(
            key=lambda x: (x.match_success_rate, x.confidence, x.discriminative_score),
            reverse=True
        )
        
        # Limit size (max 20% of total cache)
        max_high_quality = max(10, self.cache_size // 5)
        self.high_quality_cache = self.high_quality_cache[:max_high_quality]

    def _maintain_cache_size(self):
        """Cache maintenance with intelligent pruning"""
        # General cache size maintained by deque maxlen
        
        # Maintain label-specific caches with quality-based pruning
        for label in list(self.label_cache.keys()):
            label_cache = self.label_cache[label]
            if len(label_cache) > self.cache_size // 4:  # Limit per label
                # Sort by quality metrics and keep best
                label_cache.sort(
                    key=lambda x: (x.match_success_rate, x.confidence, x.last_accessed),
                    reverse=True
                )
                self.label_cache[label] = label_cache[:self.cache_size // 4]

    def _update_progress(self):
        """Update progress manager"""
        current_cache_size = len(self.shapelet_cache)
        if self.progress_manager:
            if not self._cache_task_id:
                self._cache_task_id = self.progress_manager.start_main_progress(
                    total=self.cache_size,
                    description="Shapelet Cache",
                    task_id="shapelet_cache"
                )
            self.progress_manager.update_task(
                task_id="shapelet_cache",
                advance=1 if current_cache_size > self._last_cache_size else 0,
                description=f"Cache: {current_cache_size}/{self.cache_size} entries"
            )
            self._last_cache_size = current_cache_size

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.shapelet_cache:
            return {"empty": True}
            
        recent_time = time.time() - 3600  # Last hour
        recent_entries = [s for s in self.shapelet_cache if s.timestamp > recent_time]
        
        # Calculate key performance metrics
        metrics = {
            "cache_statistics": {
                "total_cached": len(self.shapelet_cache),
                "unique_labels": len(self.label_cache),
                "high_quality_entries": len(self.high_quality_cache),
                "recent_entries": len(recent_entries),
            },
            "performance_metrics": {
                "early_match_rate": self.early_matches / max(self.total_matches, 1),
                "cache_hit_rate": self.cache_hits / max(self.total_matches, 1),
                "prediction_accuracy": self.correct_predictions / max(self.total_predictions, 1),
                "avg_match_success_rate": np.mean([s.match_success_rate for s in self.shapelet_cache]),
            },
            "quality_metrics": {
                "avg_confidence": np.mean([s.confidence for s in self.shapelet_cache]),
                "avg_discriminative_score": np.mean([s.discriminative_score for s in self.shapelet_cache]),
                "avg_usage_count": np.mean([s.usage_count for s in self.shapelet_cache]),
            },
            "adaptive_threshold": {
                "current_threshold": self.threshold_manager.get_current_threshold(),
                "initial_threshold": self.initial_threshold,
                "threshold_enabled": self.enable_adaptive_threshold,
            }
        }
        return metrics


def create_early_matcher(config: Dict[str, Any] = None, progress_manager=None) -> EarlyShapeletMatcher:
    """Factory function to create early matcher"""
    if config is None:
        config = EARLY_CONFIG
        
    return EarlyShapeletMatcher(
        early_match_threshold=config.get("early_match_threshold", 0.6),
        max_lookback=config.get("max_lookback", 10),
        min_confidence=config.get("min_confidence", 0.5),
        cache_size=config.get("cache_size", 200),
        progress_manager=progress_manager,
        enable_adaptive_threshold=config.get("enable_adaptive_threshold", True)
    )