import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

from utils.logger import get_logger, log_attention_weights

logger = get_logger(__name__)


class AttentionEvaluator:
    """Attention-based evaluation for shapelet similarity"""
    
    def __init__(self, attention_threshold: float = 0.8, permax_scale: float = 1.0):
        """ Initialize attention evaluator
        
        Args:
            attention_threshold: Threshold for attention evaluation
            permax_scale: Scaling factor for permax normalization
        """
        self.attention_threshold = attention_threshold
        self.permax_scale = permax_scale

        self.label_frequencies = defaultdict(int)
        self.label_sequences = defaultdict(lambda: deque(maxlen=100))
        self.label_timestamps = defaultdict(list)
        
        # Attention weights cache
        self.attention_weights = {}
        
        logger.info(f"Initialized AttentionEvaluator with threshold={attention_threshold}, scale={permax_scale}")
    
    def evaluate_attention(self, input_shapelet: np.ndarray, cached_shapelets: List[np.ndarray], 
                          cached_labels: List[str], similarities: List[float]) -> Dict[str, float]:
        """ Evaluate attention weights for input shapelet based on cached shapelets
        
        Args:
            input_shapelet: Input shapelet to evaluate
            cached_shapelets: List of cached shapelets
            cached_labels: List of labels corresponding to cached shapelets
            similarities: List of similarities between input and cached shapelets
            
        Returns:
            Dictionary of attention weights by label
        """
        if len(cached_shapelets) != len(cached_labels) or len(cached_shapelets) != len(similarities):
            raise ValueError("Cached shapelets, labels, and similarities must have same length")
        
        # Filter shapelets above attention threshold
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= self.attention_threshold]
        
        if not filtered_indices:
            logger.debug("No shapelets above attention threshold")
            return {}
        
        # Update frequency and sequence tracking
        self._update_tracking([cached_labels[i] for i in filtered_indices])
        
        # Calculate attention weights
        attention_weights = self._calculate_attention_weights(
            [cached_labels[i] for i in filtered_indices],
            [similarities[i] for i in filtered_indices]
        )
        
        # Apply PerMax normalization
        normalized_weights = self._permax_normalization(attention_weights)
        
        # Update cache
        self.attention_weights.update(normalized_weights)
        
        logger.debug(f"Attention evaluation completed for {len(filtered_indices)} shapelets")
        log_attention_weights(logger, normalized_weights)
        
        return normalized_weights
    
    def _update_tracking(self, labels: List[str]):
        """Update frequency and sequence tracking for labels"""
        current_time = len(self.label_timestamps)
        
        for label in labels:
            # Update frequency
            self.label_frequencies[label] += 1
            
            # Update sequence
            self.label_sequences[label].append(1)
            
            # Update timestamps
            self.label_timestamps[label].append(current_time)
        
        logger.debug(f"Updated tracking for labels: {set(labels)}")
    
    def _calculate_attention_weights(self, labels: List[str], similarities: List[float]) -> Dict[str, float]:
        """ Calculate attention weights based on frequency and recency
        
        Args:
            labels: List of labels
            similarities: List of similarities
            
        Returns:
            Raw attention weights by label
        """
        label_weights = defaultdict(list)
        
        # Group similarities by label
        for label, similarity in zip(labels, similarities):
            label_weights[label].append(similarity)
        
        # Calculate weights for each label
        attention_weights = {}
        
        for label, sims in label_weights.items():
            # Frequency component (normalized)
            frequency = len(sims)
            total_frequency = sum(self.label_frequencies.values())
            freq_component = frequency / max(total_frequency, 1)
            
            # Continuity component (average sequence length)
            sequence_length = len(self.label_sequences[label])
            continuity_component = min(sequence_length / 10.0, 1.0)  # Normalize to [0, 1]
            
            # Similarity component (average similarity)
            avg_similarity = np.mean(sims)
            
            # Combined weight
            weight = (freq_component * 0.4 + continuity_component * 0.3 + avg_similarity * 0.3)
            attention_weights[label] = weight
            
            logger.debug(f"Label {label}: freq={freq_component:.3f}, "
                       f"continuity={continuity_component:.3f}, "
                       f"similarity={avg_similarity:.3f}, "
                       f"combined={weight:.3f}")
        
        return attention_weights
    
    def _permax_normalization(self, weights: Dict[str, float]) -> Dict[str, float]:
        """ Apply PerMax normalization instead of softmax
        
        Args:
            weights: Raw attention weights
            
        Returns:
            PerMax normalized weights
        """
        if not weights:
            return {}
        
        # Apply log2 transformation (PerMax)
        permax_weights = {}
        
        for label, weight in weights.items():
            if weight > 0:
                # PerMax transformation: log2(x)
                permax_weight = math.log2(weight) * self.permax_scale
                permax_weights[label] = max(permax_weight, 0)  # Ensure non-negative
            else:
                permax_weights[label] = 0.0
        
        # Normalize to sum to 1
        total_weight = sum(permax_weights.values())
        
        if total_weight > 0:
            normalized_weights = {label: w / total_weight for label, w in permax_weights.items()}
        else:
            # Uniform distribution if all weights are zero
            n_labels = len(permax_weights)
            normalized_weights = {label: 1.0 / n_labels for label in permax_weights.keys()}
        
        logger.debug(f"PerMax normalization: {dict(weights)} -> {normalized_weights}")
        return normalized_weights
    
    def attention_score(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                       scale_factor: Optional[float] = None) -> np.ndarray:
        """ Calculate attention score using PerMax normalization
        
        Args:
            query: Query vector
            key: Key vector
            value: Value vector
            scale_factor: Scaling factor for attention
            
        Returns:
            Attention-weighted value
        """
        if scale_factor is None:
            scale_factor = 1.0 / math.sqrt(len(query))
        
        # Calculate attention scores
        attention_scores = np.dot(query, key.T) * scale_factor
        
        # Apply PerMax normalization
        permax_scores = np.log2(np.maximum(attention_scores, 1e-8)) * self.permax_scale
        permax_scores = np.maximum(permax_scores, 0)
        
        # Normalize
        if np.sum(permax_scores) > 0:
            attention_weights = permax_scores / np.sum(permax_scores)
        else:
            attention_weights = np.ones_like(permax_scores) / len(permax_scores)
        
        # Apply to values
        attention_value = np.dot(attention_weights, value)
        
        logger.debug(f"Attention score calculated: shape={attention_value.shape}")
        return attention_value
    
    def get_attention_summary(self) -> Dict[str, any]:
        """Get summary of attention evaluation statistics"""
        return {
            "total_evaluations": sum(self.label_frequencies.values()),
            "unique_labels": len(self.label_frequencies),
            "label_frequencies": dict(self.label_frequencies),
            "avg_sequence_lengths": {label: len(seq) for label, seq in self.label_sequences.items()},
            "attention_weights": self.attention_weights.copy()
        }
    
    def reset_attention(self):
        """Reset attention tracking"""
        self.label_frequencies.clear()
        self.label_sequences.clear()
        self.label_timestamps.clear()
        self.attention_weights.clear()
        
        logger.info("Reset attention evaluator")
    
    def get_top_attention_labels(self, k: int = 5) -> List[Tuple[str, float]]:
        """ Get top k labels by attention weight
        
        Args:
            k: Number of top labels to return
            
        Returns:
            List of (label, weight) tuples
        """
        if not self.attention_weights:
            return []
        
        sorted_weights = sorted(self.attention_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_weights[:k]


class StreamingAttentionEvaluator(AttentionEvaluator):
    """Streaming version of attention evaluator for continuous data"""
    
    def __init__(self, attention_threshold: float = 0.8, permax_scale: float = 1.0, 
                 window_size: int = 100, decay_factor: float = 0.95):
        """ Initialize streaming attention evaluator
        
        Args:
            attention_threshold: Threshold for attention evaluation
            permax_scale: Scaling factor for permax normalization
            window_size: Size of sliding window for recent data
            decay_factor: Decay factor for old data
        """
        super().__init__(attention_threshold, permax_scale)
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # Sliding windows for streaming data
        self.recent_labels = deque(maxlen=window_size)
        self.recent_similarities = deque(maxlen=window_size)
        self.recent_timestamps = deque(maxlen=window_size)
        
        logger.info(f"Initialized StreamingAttentionEvaluator with window_size={window_size}")
    
    def evaluate_streaming_attention(self, new_shapelet: np.ndarray, new_label: str, 
                                   new_similarity: float) -> Dict[str, float]:
        """ Evaluate attention for streaming data
        
        Args:
            new_shapelet: New shapelet
            new_label: New label
            new_similarity: New similarity score
            
        Returns:
            Updated attention weights
        """
        # Add to recent data
        self.recent_labels.append(new_label)
        self.recent_similarities.append(new_similarity)
        self.recent_timestamps.append(len(self.recent_timestamps))
        
        # Apply decay to old frequencies
        self._apply_decay()
        
        # Evaluate attention based on recent window
        if new_similarity >= self.attention_threshold:
            attention_weights = self._calculate_attention_weights(
                list(self.recent_labels), list(self.recent_similarities)
            )
            normalized_weights = self._permax_normalization(attention_weights)
            self.attention_weights.update(normalized_weights)
            
            logger.debug(f"Streaming attention updated for label {new_label}")
            return normalized_weights
        
        return {}
    
    def _apply_decay(self):
        """Apply decay to old frequency counts"""
        for label in list(self.label_frequencies.keys()):
            self.label_frequencies[label] *= self.decay_factor
            
            # Remove labels with very low frequency
            if self.label_frequencies[label] < 0.01:
                del self.label_frequencies[label]
                if label in self.label_sequences:
                    del self.label_sequences[label]
                if label in self.label_timestamps:
                    del self.label_timestamps[label]
        
        logger.debug(f"Applied decay with factor {self.decay_factor}")