import numpy as np
from typing import Optional
from fastdtw import fastdtw
from scipy.spatial.distance import minkowski
from utils.logger import get_logger

logger = get_logger(__name__)


class ShapeletSimilarityCalculator:
    """Calculates similarities between shapelets using various distance metrics"""
    
    def __init__(self, distance_threshold: float = 0.1, similarity_threshold: float = 0.9):
        """ Initialize similarity calculator
        
        Args:
            distance_threshold: Threshold for mixed Minkowski distance
            similarity_threshold: Threshold for similarity caching
        """
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.minkowski_cache = {}  # Cache for optimal p values
        
        logger.info(f"Initialized ShapeletSimilarityCalculator with distance_threshold={distance_threshold}, similarity_threshold={similarity_threshold}")
    
    def calculate_distance(self, shapelet1: np.ndarray, shapelet2: np.ndarray, 
                          p: Optional[int] = None) -> float:
        """ Calculate distance between two shapelets
        
        Args:
            shapelet1: First shapelet array
            shapelet2: Second shapelet array
            p: Minkowski distance parameter (if None, auto-select optimal)
            
        Returns:
            Distance value
        """
        if len(shapelet1) == len(shapelet2):
            # Equal length: use Minkowski distance with optimal p
            return self._minkowski_distance(shapelet1, shapelet2, p)
        else:
            # Unequal length: use FastDTW
            return self._fastdtw_distance(shapelet1, shapelet2)
    
    def _minkowski_distance(self, shapelet1: np.ndarray, shapelet2: np.ndarray, 
                           p: Optional[int] = None) -> float:
        """ Calculate Minkowski distance with optimal p selection
        
        Args:
            shapelet1: First shapelet array
            shapelet2: Second shapelet array
            p: Minkowski parameter (if None, find optimal)
            
        Returns:
            Minkowski distance
        """
        shapelet_pair = (tuple(shapelet1), tuple(shapelet2))
        
        if p is None:
            # Check cache for optimal p
            if shapelet_pair in self.minkowski_cache:
                p = self.minkowski_cache[shapelet_pair]
            else:
                # Find optimal p using mixed Minkowski strategy
                p = self._find_optimal_p(shapelet1, shapelet2)
                self.minkowski_cache[shapelet_pair] = p
        
        try:
            distance = minkowski(shapelet1, shapelet2, p)
            logger.debug(f"Minkowski distance (p={p}): {distance:.4f}")
            return distance
        except Exception as e:
            logger.error(f"Error calculating Minkowski distance: {e}")
            return float('inf')
    
    def _find_optimal_p(self, shapelet1: np.ndarray, shapelet2: np.ndarray) -> int:
        """ Find optimal p value for Minkowski distance using mixed strategy
        
        Args:
            shapelet1: First shapelet array
            shapelet2: Second shapelet array
            
        Returns:
            Optimal p value (1 or 2)
        """
        # Try p=1 and p=2, select based on distance threshold
        distances = {}
        
        for p in [1, 2]:
            try:
                dist = minkowski(shapelet1, shapelet2, p)
                distances[p] = dist
            except Exception as e:
                logger.warning(f"Error with p={p}: {e}")
                distances[p] = float('inf')
        
        # Select p that gives distance > threshold
        for p in [1, 2]:
            if distances[p] > self.distance_threshold:
                logger.debug(f"Selected p={p} with distance={distances[p]:.4f}")
                return p
        
        # If both distances <= threshold, return p=2 (Euclidean)
        logger.debug(f"Default to p=2, distances: {distances}")
        return 2
    
    def _fastdtw_distance(self, shapelet1: np.ndarray, shapelet2: np.ndarray) -> float:
        """ Calculate FastDTW distance for unequal length shapelets
        
        Args:
            shapelet1: First shapelet array
            shapelet2: Second shapelet array
            
        Returns:
            FastDTW distance
        """
        try:
            distance, _ = fastdtw(shapelet1, shapelet2)
            logger.debug(f"FastDTW distance: {distance:.4f}")
            return distance
        except Exception as e:
            logger.error(f"Error calculating FastDTW distance: {e}")
            return float('inf')
    
    def calculate_similarity(self, shapelet1: np.ndarray, shapelet2: np.ndarray, 
                            p: Optional[int] = None) -> float:
        """ Calculate similarity between two shapelets
        
        Args:
            shapelet1: First shapelet array
            shapelet2: Second shapelet array
            p: Minkowski distance parameter (optional)
            
        Returns:
            Similarity value (0 to 1)
        """
        distance = self.calculate_distance(shapelet1, shapelet2, p)
        similarity = 1.0 / (1.0 + distance)
        
        logger.debug(f"Similarity: {similarity:.4f} (distance: {distance:.4f})")
        return similarity
    
    def calculate_similarity_matrix(self, shapelets: list) -> np.ndarray:
        """ Calculate pairwise similarity matrix for shapelets
        
        Args:
            shapelets: List of shapelet arrays
            
        Returns:
            Similarity matrix
        """
        n_shapelets = len(shapelets)
        similarity_matrix = np.zeros((n_shapelets, n_shapelets))
        
        logger.info(f"Calculating similarity matrix for {n_shapelets} shapelets")
        
        for i in range(n_shapelets):
            for j in range(i, n_shapelets):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self.calculate_similarity(shapelets[i], shapelets[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        logger.info(f"Similarity matrix calculation completed")
        return similarity_matrix
    
    def get_ranking(self, shapelet: np.ndarray, shapelets: list) -> list:
        """ Get ranking of shapelets by similarity to reference shapelet
        
        Args:
            shapelet: Reference shapelet
            shapelets: List of shapelets to rank
            
        Returns:
            List of (index, similarity) tuples sorted by similarity (descending)
        """
        similarities = []
        
        for i, s in enumerate(shapelets):
            similarity = self.calculate_similarity(shapelet, s)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Ranking calculated for shapelet, top similarity: {similarities[0][1]:.4f}")
        return similarities
    
    def clear_cache(self):
        """Clear Minkowski parameter cache"""
        self.minkowski_cache.clear()
        logger.debug("Cleared Minkowski parameter cache")


# Utility functions
def normalize_shapelet(shapelet: np.ndarray) -> np.ndarray:
    """Normalize shapelet to zero mean and unit variance"""
    if np.std(shapelet) == 0:
        return np.zeros_like(shapelet)
    return (shapelet - np.mean(shapelet)) / np.std(shapelet)


def z_normalize_shapelet(shapelet: np.ndarray) -> np.ndarray:
    """Z-normalize shapelet"""
    return normalize_shapelet(shapelet)


def min_max_normalize_shapelet(shapelet: np.ndarray) -> np.ndarray:
    """Min-max normalize shapelet to [0, 1] range"""
    min_val = np.min(shapelet)
    max_val = np.max(shapelet)
    
    if max_val == min_val:
        return np.zeros_like(shapelet)
    
    return (shapelet - min_val) / (max_val - min_val)