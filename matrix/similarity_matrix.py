import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from utils.logger import get_logger, log_matrix_operation
from matrix.shapelet_similarity import ShapeletSimilarityCalculator
from matrix.attention_evaluator import AttentionEvaluator

logger = get_logger(__name__)


class SimilarityMatrix:
    """Shapelet similarity matrix with incremental updates and ranking storage"""
    
    def __init__(self, size: int = None, distance_threshold: float = 0.1, 
                 similarity_threshold: float = 0.9, matrix_logs_dir: str = "./matrix_logs",
                 auto_expand: bool = True, expand_factor: float = 1.5, max_size: int = 2000):
        """ Initialize similarity matrix
        
        Args:
            size: Initial size of the matrix (K)
            distance_threshold: Threshold for distance calculations
            similarity_threshold: Threshold for similarity caching
            matrix_logs_dir: Directory for matrix snapshots
            auto_expand: Whether to automatically expand matrix when full
            expand_factor: Factor to expand matrix size (1.5 = 50% increase)
            max_size: Maximum allowed matrix size
        """
        # Set default size if None provided
        self.initial_size = size if size is not None else 100
        self.size = self.initial_size
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.matrix_logs_dir = Path(matrix_logs_dir)
        self.matrix_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic expansion settings
        self.auto_expand = auto_expand
        self.expand_factor = expand_factor
        self.max_size = max_size
        self.expansion_count = 0
        
        # Initialize matrix with zeros - ensure proper shape
        self.matrix = np.zeros((self.size, self.size), dtype=np.float64)
        self.shapelet_count = 0
        
        # Ranking storage: rank -> (i, j) index pairs
        self.ranking_map = {}
        
        # Similarity calculator and attention evaluator
        self.similarity_calculator = ShapeletSimilarityCalculator(
            distance_threshold=distance_threshold,
            similarity_threshold=similarity_threshold
        )
        self.attention_evaluator = AttentionEvaluator(
            attention_threshold=similarity_threshold
        )
        
        # Matrix evolution tracking
        self.snapshot_counter = 0
        self.matrix_history = []
        
        # Shapelet storage for proper similarity calculations
        self._shapelet_storage = {}  # index -> (shapelet, label)
        self._max_shapelet_length = 0
        
        logger.info(f"Initialized SimilarityMatrix with size={self.size}")

    def initialize_matrix(self, shapelets: List[np.ndarray], labels: List[str]):
        """ Initialize matrix with initial shapelets
        
        Args:
            shapelets: List of initial shapelets
            labels: List of corresponding labels
        """
        if len(shapelets) != len(labels):
            raise ValueError("Shapelets and labels must have same length")
        
        if len(shapelets) == 0:
            logger.warning("Empty shapelet list provided for initialization")
            return
        
        # Check if we need to expand matrix before initialization
        if len(shapelets) > self.size:
            if self.auto_expand:
                self._expand_matrix_to_fit(len(shapelets))
            else:
                raise ValueError(f"Number of shapelets ({len(shapelets)}) exceeds matrix size ({self.size})")
        
        # Store shapelets first
        self.shapelet_count = len(shapelets)
        for i, (shapelet, label) in enumerate(zip(shapelets, labels)):
            self._store_shapelet(i, shapelet, label)
        
        # Calculate pairwise similarities
        logger.info(f"Initializing matrix with {self.shapelet_count} shapelets (matrix size: {self.size})")
        
        for i in range(self.shapelet_count):
            for j in range(self.shapelet_count):
                if i == j:
                    self.matrix[i, j] = 0.0  # Diagonal is 0
                elif i == 0:
                    # First row: store ranking of S1 vs Sj
                    try:
                        ranking = self._calculate_ranking(shapelets[0], shapelets[j])
                        self.matrix[i, j] = ranking
                    except Exception as e:
                        logger.warning(f"Error calculating ranking for (0,{j}): {e}")
                        self.matrix[i, j] = -1.0
                elif j == 0:
                    # First column: store similarity of Si vs S1
                    try:
                        similarity = self.similarity_calculator.calculate_similarity(
                            shapelets[i], shapelets[0]
                        )
                        self.matrix[i, j] = similarity
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for ({i},0): {e}")
                        self.matrix[i, j] = -1.0
                else:
                    # Other positions: dynamic calculation or placeholder
                    self.matrix[i, j] = -1.0  # Placeholder for dynamic calculation
        
        # Build ranking map
        self._build_ranking_map(shapelets)
        
        # Record initial snapshot
        self._record_snapshot("initialization", {
            "shapelets": len(shapelets),
            "labels": len(set(labels)),
            "matrix_size": self.size,
            "avg_shapelet_length": np.mean([len(s) for s in shapelets])
        })
        
        logger.info("Matrix initialization completed")

    def _expand_matrix_to_fit(self, required_size: int):
        """ Expand matrix to accommodate required number of shapelets
        
        Args:
            required_size: Minimum size needed
        """
        if required_size > self.max_size:
            raise ValueError(f"Required size ({required_size}) exceeds maximum allowed size ({self.max_size})")
        
        # Calculate new size with proper bounds checking
        new_size = int(max(required_size, int(self.size * self.expand_factor)))
        new_size = min(new_size, self.max_size)  # Don't exceed max_size
        
        logger.info(f"Expanding matrix from {self.size} to {new_size} (required: {required_size})")

        new_matrix = np.zeros((new_size, new_size), dtype=np.float64)

        if self.shapelet_count > 0 and self.size > 0:
            copy_size = min(self.size, new_size)
            new_matrix[:copy_size, :copy_size] = self.matrix[:copy_size, :copy_size]

        self.matrix = new_matrix
        self.size = new_size
        self.expansion_count += 1
        
        logger.info(f"Matrix expansion completed. Expansion count: {self.expansion_count}")

        if self.shapelet_count > 0:
            logger.info("Preserving valid ranking map entries after expansion")
            valid_rankings = {}
            for rank_idx, (i, j) in self.ranking_map.items():
                # Only keep entries where both indices are within the current shapelet count
                if i < self.shapelet_count and j < self.shapelet_count:
                    valid_rankings[rank_idx] = (i, j)
            self.ranking_map = valid_rankings
            logger.info(f"Preserved {len(self.ranking_map)} valid ranking entries")

    def _calculate_ranking(self, reference: np.ndarray, target: np.ndarray) -> float:
        """ Calculate ranking score between reference and target shapelets
        
        Args:
            reference: Reference shapelet
            target: Target shapelet
            
        Returns:
            Ranking score (higher means more similar)
        """
        similarity = self.similarity_calculator.calculate_similarity(reference, target)
        # Convert similarity to ranking score (inverse relationship)
        ranking_score = 1.0 / (1.0 + (1.0 - similarity))
        return ranking_score

    def _build_ranking_map(self, shapelets: List[np.ndarray]):
        """Build ranking map for efficient lookup"""
        logger.info("Building ranking map")
        
        # Calculate all pairwise rankings
        all_rankings = []
        for i in range(self.shapelet_count):
            for j in range(self.shapelet_count):
                if i != j:
                    ranking = self.matrix[i, j] if self.matrix[i, j] >= 0 else \
                              self._calculate_ranking(shapelets[i], shapelets[j])
                    all_rankings.append((ranking, (i, j)))
        
        # Sort by ranking score (descending)
        all_rankings.sort(key=lambda x: x[0], reverse=True)
        
        # Build ranking map
        for rank_idx, (score, (i, j)) in enumerate(all_rankings):
            self.ranking_map[rank_idx] = (i, j)
        
        logger.info(f"Built ranking map with {len(self.ranking_map)} entries")

    def update_matrix(self, new_shapelet: np.ndarray, new_label: str, 
                     attention_weights: Dict[str, float]) -> bool:
        """ Incrementally update matrix based on attention evaluation
        
        Args:
            new_shapelet: New shapelet to add
            new_label: Label for the new shapelet
            attention_weights: Attention weights from evaluation
            
        Returns:
            True if update was successful
        """
        # Validate input
        if not isinstance(new_shapelet, np.ndarray) or new_shapelet.size == 0:
            logger.error("Invalid shapelet provided for matrix update")
            return False
        
        # Check if expansion is needed before attempting to add
        if self.shapelet_count >= self.size:
            if self.auto_expand:
                try:
                    self._expand_matrix_to_fit(self.shapelet_count + 1)
                except ValueError as e:
                    logger.error(f"Cannot expand matrix: {e}")
                    return False
            else:
                logger.warning("Matrix is full, cannot add more shapelets")
                return False
        
        # Add new shapelet to matrix
        new_idx = self.shapelet_count
        self.shapelet_count += 1
        
        # Store shapelet for future reference (fix the placeholder issue)
        self._store_shapelet(new_idx, new_shapelet, new_label)
        
        # Calculate similarities with existing shapelets
        logger.info(f"Updating matrix for new shapelet at index {new_idx} "
                   f"(matrix size: {self.size}, shapelet_count: {self.shapelet_count})")

        self.matrix[new_idx, new_idx] = 0.0
        
        for existing_idx in range(new_idx):
            existing_shapelet = self._get_shapelet_at(existing_idx)
            if existing_shapelet is None or existing_shapelet.size == 0:
                logger.warning(f"Skipping similarity calculation for invalid shapelet at index {existing_idx}")
                self.matrix[new_idx, existing_idx] = -1.0
                self.matrix[existing_idx, new_idx] = -1.0
                continue
            
            try:
                # Calculate similarity
                similarity = self.similarity_calculator.calculate_similarity(
                    new_shapelet, existing_shapelet
                )
                
                # Validate similarity value
                if not np.isfinite(similarity):
                    logger.warning(f"Non-finite similarity calculated for indices {new_idx},{existing_idx}")
                    similarity = -1.0
                
                # Update matrix symmetrically
                self.matrix[new_idx, existing_idx] = similarity
                self.matrix[existing_idx, new_idx] = similarity
                
                # Update first row/column if necessary
                if existing_idx == 0:
                    ranking = self._calculate_ranking(existing_shapelet, new_shapelet)
                    if np.isfinite(ranking):
                        self.matrix[0, new_idx] = ranking
                        self.matrix[new_idx, 0] = similarity
            
            except Exception as e:
                logger.error(f"Error calculating similarity between {new_idx} and {existing_idx}: {e}")
                self.matrix[new_idx, existing_idx] = -1.0
                self.matrix[existing_idx, new_idx] = -1.0
        
        # Update ranking map
        self._update_ranking_map(new_idx, new_shapelet)
        
        # Record snapshot
        self._record_snapshot("incremental_update", {
            "new_shapelet_idx": new_idx,
            "new_label": new_label,
            "attention_weights": attention_weights,
            "matrix_shape": (self.shapelet_count, self.shapelet_count),
            "shapelet_length": len(new_shapelet)
        })
        
        logger.info(f"Matrix update completed for shapelet {new_idx}")
        return True

    def _store_shapelet(self, idx: int, shapelet: np.ndarray, label: str):
        """Store shapelet for future reference"""
        self._shapelet_storage[idx] = (shapelet.copy(), label)
        self._max_shapelet_length = max(self._max_shapelet_length, len(shapelet))
        logger.debug(f"Stored shapelet {idx} (length: {len(shapelet)}, label: {label})")

    def _get_shapelet_at(self, idx: int) -> Optional[np.ndarray]:
        """Get shapelet at specified index"""
        if idx in self._shapelet_storage:
            return self._shapelet_storage[idx][0]
        else:
            logger.warning(f"Shapelet at index {idx} not found in storage")
            return None

    def _update_ranking_map(self, new_idx: int, new_shapelet: np.ndarray):
        """Update ranking map with new shapelet"""
        logger.info(f"Updating ranking map for shapelet {new_idx}")
        
        # Calculate new rankings
        new_rankings = []
        for existing_idx in range(new_idx):
            if existing_idx == 0:
                # Update ranking with reference shapelet
                ranking = self.matrix[0, new_idx]
                new_rankings.append((ranking, (0, new_idx)))
            else:
                # Calculate ranking for other pairs
                ranking = self._calculate_ranking(
                    self._get_shapelet_at(existing_idx), new_shapelet
                )
                new_rankings.append((ranking, (existing_idx, new_idx)))
        
        # Update ranking map
        current_rankings = [(self.matrix[i, j], (i, j)) for (i, j) in self.ranking_map.values()]
        current_rankings.extend(new_rankings)
        
        # Sort and rebuild map
        current_rankings.sort(key=lambda x: x[0], reverse=True)
        self.ranking_map.clear()
        
        for rank_idx, (score, (i, j)) in enumerate(current_rankings):
            self.ranking_map[rank_idx] = (i, j)
        
        logger.info(f"Ranking map updated with {len(self.ranking_map)} entries")

    def get_top_k_similar_pairs(self, k: int = 10) -> List[Tuple[Tuple[int, int], float]]:
        """ Get top K most similar shapelet pairs
        
        Args:
            k: Number of top pairs to return
            
        Returns:
            List of ((i, j), similarity) tuples
        """
        if k <= 0:
            return []
        
        # Get similarities from matrix
        similarities = []
        for rank_idx in range(min(k, len(self.ranking_map))):
            if rank_idx in self.ranking_map:
                (i, j) = self.ranking_map[rank_idx]
                if i < self.shapelet_count and j < self.shapelet_count:
                    similarity = self.matrix[i, j]
                    if similarity >= 0:  # Valid similarity
                        similarities.append(((i, j), similarity))
        
        logger.info(f"Retrieved top {len(similarities)} similar pairs")
        return similarities

    def update_with_shapelets(self, shapelets: List[np.ndarray]) -> bool:
        """ Update matrix with multiple shapelets at once
        
        Args:
            shapelets: List of shapelets to add to the matrix
            
        Returns:
            True if update was successful
        """
        logger.info(f"Updating matrix with {len(shapelets)} shapelets")
        
        if not shapelets:
            logger.warning("Empty shapelet list provided for matrix update")
            return False
        
        try:
            labels = [f"shapelet_{i}" for i in range(len(shapelets))]
            self.reset_matrix()
            self.initialize_matrix(shapelets, labels)
            
            logger.info(f"Successfully updated matrix with {len(shapelets)} shapelets")
            return True
            
        except Exception as e:
            logger.error(f"Error updating matrix with shapelets: {e}")
            return False

    def get_matrix_statistics(self) -> Dict[str, Any]:
        """Get statistics about the similarity matrix"""
        if self.shapelet_count == 0:
            return {"empty": True}
        
        # Get valid similarities (excluding placeholders and diagonal)
        valid_similarities = []
        for i in range(self.shapelet_count):
            for j in range(i + 1, self.shapelet_count):
                similarity = self.matrix[i, j]
                if similarity >= 0:  # Valid similarity
                    valid_similarities.append(similarity)
        
        if not valid_similarities:
            return {"empty": True}
        
        stats = {
            "shapelet_count": self.shapelet_count,
            "matrix_size": self.size,
            "valid_pairs": len(valid_similarities),
            "mean_similarity": np.mean(valid_similarities),
            "std_similarity": np.std(valid_similarities),
            "min_similarity": np.min(valid_similarities),
            "max_similarity": np.max(valid_similarities),
            "median_similarity": np.median(valid_similarities),
            "ranking_map_size": len(self.ranking_map)
        }
        
        logger.debug(f"Matrix statistics: {stats}")
        return stats

    def _record_snapshot(self, operation: str, metadata: Dict[str, Any]):
        """ Record matrix snapshot for evolution tracking
        
        Args:
            operation: Operation that triggered the snapshot
            metadata: Additional metadata about the operation
        """
        self.snapshot_counter += 1
        
        # Ensure consistent matrix shape for snapshot
        actual_matrix_shape = self.matrix.shape
        effective_matrix_data = self.matrix[:self.shapelet_count, :self.shapelet_count].copy()
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "snapshot_id": self.snapshot_counter,
            "matrix_shape": (self.shapelet_count, self.shapelet_count),
            "matrix_size": actual_matrix_shape[0],  # Store full matrix size
            "matrix_data": effective_matrix_data,
            "ranking_map": self.ranking_map.copy(),
            "metadata": metadata
        }
        
        # Save to file
        snapshot_file = self.matrix_logs_dir / f"snapshot_{self.snapshot_counter:03d}.npz"
        
        try:
            # Convert ranking_map to a format that ensures consistent shape
            ranking_items = list(snapshot["ranking_map"].items())
            
            # Validate and sanitize ranking data before serialization
            validated_ranking_items = []
            for rank, indices in ranking_items:
                try:
                    # Ensure rank is integer
                    rank_int = int(rank)
                    
                    # Ensure indices is a valid 2-element tuple
                    if isinstance(indices, (tuple, list)) and len(indices) == 2:
                        # Valid 2-element index pair
                        validated_indices = (int(indices[0]), int(indices[1]))
                        validated_ranking_items.append((rank_int, validated_indices))
                    elif isinstance(indices, (tuple, list)) and len(indices) == 1:
                        # Single element - convert to pair with same index
                        validated_indices = (int(indices[0]), int(indices[0]))
                        validated_ranking_items.append((rank_int, validated_indices))
                    elif isinstance(indices, (int, float, np.number)):
                        # Single number - convert to pair with same index
                        validated_indices = (int(indices), int(indices))
                        validated_ranking_items.append((rank_int, validated_indices))
                    else:
                        # Invalid format - skip this entry
                        logger.warning(f"Skipping invalid ranking entry: rank={rank}, indices={indices}")
                        continue
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Skipping corrupted ranking entry: rank={rank}, indices={indices}, error={e}")
                    continue
            
            if validated_ranking_items:
                # Convert to serializable format - use separate arrays for ranks and indices
                ranks = np.array([item[0] for item in validated_ranking_items], dtype=np.int32)
                indices_array = np.array([item[1] for item in validated_ranking_items], dtype=np.int32)
                
                # Ensure indices_array has correct shape (N, 2)
                if indices_array.ndim == 1:
                    # Handle 1D array case - reshape to 2D
                    if len(indices_array) % 2 == 0:
                        indices_array = indices_array.reshape(-1, 2)
                        logger.warning(f"Reshaped 1D indices array to 2D: {indices_array.shape}")
                    else:
                        logger.error(f"Cannot reshape 1D indices array with odd length: {len(indices_array)}")
                        # Create empty arrays as fallback
                        ranking_data = {
                            'ranks': np.array([], dtype=np.int32),
                            'indices': np.zeros((0, 2), dtype=np.int32)
                        }
                elif indices_array.ndim == 2 and indices_array.shape[1] == 2:
                    # Correct 2D format
                    ranking_data = {
                        'ranks': ranks,
                        'indices': indices_array
                    }
                else:
                    logger.error(f"Invalid indices array shape: {indices_array.shape}")
                    # Create empty arrays as fallback
                    ranking_data = {
                        'ranks': np.array([], dtype=np.int32),
                        'indices': np.zeros((0, 2), dtype=np.int32)
                    }
            else:
                ranking_data = {
                    'ranks': np.array([], dtype=np.int32),
                    'indices': np.zeros((0, 2), dtype=np.int32)
                }
            
            logger.info(f"Validated ranking data: {len(validated_ranking_items)} valid entries out of {len(ranking_items)} total")
            
            np.savez_compressed(
                snapshot_file,
                matrix=snapshot["matrix_data"],
                matrix_full_size=snapshot["matrix_size"],
                ranks=ranking_data['ranks'],
                indices=ranking_data['indices'],
                metadata=json.dumps({
                    "timestamp": snapshot["timestamp"],
                    "operation": snapshot["operation"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "matrix_shape": snapshot["matrix_shape"],
                    "matrix_size": snapshot["matrix_size"],
                    "metadata": snapshot["metadata"]
                })
            )
            
            self.matrix_history.append(snapshot)
            log_matrix_operation(logger, operation, snapshot["matrix_shape"], snapshot["timestamp"])
            logger.info(f"Recorded snapshot {self.snapshot_counter} to {snapshot_file} "
                       f"(shape: {snapshot['matrix_shape']}, full_size: {snapshot['matrix_size']})")
            
        except Exception as e:
            logger.error(f"Error recording snapshot: {e}")
            raise

    def load_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """ Load a specific snapshot
        
        Args:
            snapshot_id: ID of snapshot to load
            
        Returns:
            Snapshot data or None if not found
        """
        snapshot_file = self.matrix_logs_dir / f"snapshot_{snapshot_id:03d}.npz"
        
        if not snapshot_file.exists():
            logger.warning(f"Snapshot {snapshot_id} not found")
            return None
        
        try:
            data = np.load(snapshot_file, allow_pickle=True)
            metadata = json.loads(data["metadata"].item())
            
            # Load ranking map with proper handling of new format
            ranks = data["ranks"]
            indices = data["indices"]
            
            ranking_map = {}
            if len(ranks) > 0 and len(indices) > 0:
                try:
                    # Validate input arrays
                    if not isinstance(ranks, np.ndarray) or not isinstance(indices, np.ndarray):
                        raise ValueError("Invalid data types for ranks or indices")
                    
                    # Handle different indices formats with robust validation
                    if indices.ndim == 2 and indices.shape[1] == 2:
                        # Correct 2D format: indices is 2D array of shape (N, 2)
                        logger.info(f"Loading snapshot with 2D indices format: {indices.shape}")
                        for i, (rank, idx_pair) in enumerate(zip(ranks, indices)):
                            # Validate the index pair
                            if len(idx_pair) == 2:
                                ranking_map[int(rank)] = (int(idx_pair[0]), int(idx_pair[1]))
                            else:
                                logger.warning(f"Invalid index pair at rank {rank}: {idx_pair}")
                    elif indices.ndim == 1:
                        # 1D format - attempt to reconstruct as pairs
                        logger.warning(f"Snapshot {snapshot_id} has 1D indices format, attempting reconstruction")
                        if len(ranks) * 2 == len(indices):
                            # Assume indices are stored as flattened pairs
                            indices_2d = indices.reshape(-1, 2)
                            for i, (rank, idx_pair) in enumerate(zip(ranks, indices_2d)):
                                ranking_map[int(rank)] = (int(idx_pair[0]), int(idx_pair[1]))
                        elif len(ranks) == len(indices):
                            # Single indices - convert to pairs with same value
                            for i, (rank, idx) in enumerate(zip(ranks, indices)):
                                ranking_map[int(rank)] = (int(idx), int(idx))
                        else:
                            logger.error(f"Incompatible ranks ({len(ranks)}) and indices ({len(indices)}) lengths")
                    elif indices.ndim == 2 and indices.shape[1] != 2:
                        # Wrong number of columns - try to fix
                        logger.warning(f"Snapshot {snapshot_id} has wrong indices shape: {indices.shape}")
                        if indices.shape[1] > 2:
                            # Take first 2 columns
                            for i, (rank, idx_row) in enumerate(zip(ranks, indices)):
                                ranking_map[int(rank)] = (int(idx_row[0]), int(idx_row[1]))
                        elif indices.shape[1] == 1:
                            # Single column - duplicate values
                            for i, (rank, idx_col) in enumerate(zip(ranks, indices)):
                                ranking_map[int(rank)] = (int(idx_col[0]), int(idx_col[0]))
                        else:
                            logger.error(f"Cannot handle indices shape: {indices.shape}")
                    else:
                        logger.error(f"Unsupported indices format: ndim={indices.ndim}, shape={indices.shape}")
                
                except (ValueError, IndexError, TypeError) as e:
                    logger.error(f"Error reconstructing ranking map from snapshot {snapshot_id}: {e}")
                    # Continue with empty ranking map
                    ranking_map = {}
            
            # Get matrix data and handle potential shape mismatches
            matrix_data = data["matrix"]
            matrix_shape = metadata.get("matrix_shape", matrix_data.shape)
            matrix_full_size = metadata.get("matrix_size", max(matrix_shape))
            
            snapshot = {
                "matrix_data": matrix_data,
                "ranking_map": ranking_map,
                "metadata": metadata,
                "timestamp": metadata["timestamp"],
                "operation": metadata["operation"],
                "matrix_shape": matrix_shape,
                "matrix_full_size": matrix_full_size
            }
            
            logger.info(f"Loaded snapshot {snapshot_id} "
                       f"(shape: {matrix_shape}, full_size: {matrix_full_size})")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            return None

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get matrix evolution history"""
        return self.matrix_history.copy()

    def clear_history(self):
        """Clear matrix history and snapshots"""
        self.matrix_history.clear()
        self.snapshot_counter = 0
        
        # Delete snapshot files
        for snapshot_file in self.matrix_logs_dir.glob("snapshot_*.npz"):
            try:
                snapshot_file.unlink()
            except Exception as e:
                logger.warning(f"Error deleting snapshot file {snapshot_file}: {e}")
        
        logger.info("Cleared matrix history and snapshots")

    def reset_matrix(self):
        """Reset matrix to initial state"""
        self.matrix.fill(0.0)
        self.shapelet_count = 0
        self.ranking_map.clear()
        self.attention_evaluator.reset_attention()
        logger.info("Reset similarity matrix")

    def export_matrix(self, filepath: str, format: str = "npz"):
        """ Export matrix to file
        
        Args:
            filepath: Output file path
            format: Export format ("npz", "csv", "json")
        """
        filepath = Path(filepath)
        
        try:
            if format == "npz":
                np.savez_compressed(
                    filepath,
                    matrix=self.matrix,
                    ranking_map=np.array(list(self.ranking_map.items())),
                    shapelet_count=self.shapelet_count
                )
            elif format == "csv":
                np.savetxt(filepath, self.matrix, delimiter=",")
            elif format == "json":
                import json
                data = {
                    "matrix": self.matrix.tolist(),
                    "ranking_map": {str(k): v for k, v in self.ranking_map.items()},
                    "shapelet_count": self.shapelet_count
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported matrix to {filepath} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting matrix: {e}")
            raise

    def _validate_ranking_map_entry(self, rank: int, indices: Any) -> Optional[Tuple[int, Tuple[int, int]]]:
        """ Validate and normalize a ranking map entry
        
        Args:
            rank: The rank value
            indices: The indices (should be a 2-element tuple)
            
        Returns:
            Validated (rank, indices) tuple or None if invalid
        """
        try:
            # Ensure rank is integer
            rank_int = int(rank)
            
            # Handle different indices formats
            if isinstance(indices, (tuple, list)):
                if len(indices) == 2:
                    # Valid 2-element pair
                    return (rank_int, (int(indices[0]), int(indices[1])))
                elif len(indices) == 1:
                    # Single element - convert to pair with same index
                    idx = int(indices[0])
                    return (rank_int, (idx, idx))
                else:
                    logger.warning(f"Invalid indices length: {len(indices)}, expected 1 or 2")
                    return None
            elif isinstance(indices, (int, float, np.number)):
                # Single number - convert to pair with same index
                idx = int(indices)
                return (rank_int, (idx, idx))
            else:
                logger.warning(f"Invalid indices type: {type(indices)}, expected tuple, list, or number")
                return None
        
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Invalid ranking map entry: rank={rank}, indices={indices}, error={e}")
            return None

    def _sanitize_ranking_map(self, ranking_map: Dict[Any, Any]) -> Dict[int, Tuple[int, int]]:
        """ Sanitize ranking map by validating and normalizing all entries
        
        Args:
            ranking_map: Original ranking map
            
        Returns:
            Sanitized ranking map with only valid entries
        """
        sanitized_map = {}
        invalid_entries = 0
        
        for rank, indices in ranking_map.items():
            validated_entry = self._validate_ranking_map_entry(rank, indices)
            if validated_entry is not None:
                validated_rank, validated_indices = validated_entry
                sanitized_map[validated_rank] = validated_indices
            else:
                invalid_entries += 1
        
        if invalid_entries > 0:
            logger.warning(f"Removed {invalid_entries} invalid entries from ranking map")
        
        return sanitized_map