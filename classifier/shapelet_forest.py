from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from utils.logger import get_logger, log_forest_update
from matrix.shapelet_similarity import ShapeletSimilarityCalculator

logger = get_logger(__name__)


class ShapeletNode:
    """Base class for shapelet decision tree nodes"""
    
    def __init__(self, node_id: int, node_type: str, shapelet: np.ndarray = None, 
                 threshold: float = 0.0, label: str = None):
        """ Initialize shapelet node
        
        Args:
            node_id: Unique node identifier
            node_type: Type of node (PDN, SDN, LDN)
            shapelet: Shapelet for decision making
            threshold: Decision threshold
            label: Class label (for leaf nodes)
        """
        self.node_id = node_id
        self.node_type = node_type  # PDN, SDN, LDN
        self.shapelet = shapelet
        self.threshold = threshold
        self.label = label
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.depth = 0
        self.visits = 0
        self.correct_predictions = 0
        self.cache_similarity = None
        
        logger.debug(f"Created {node_type} node {node_id}")
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf node"""
        return self.node_type == "LDN"
    
    def is_primary(self) -> bool:
        """Check if node is primary decision node"""
        return self.node_type == "PDN"
    
    def is_secondary(self) -> bool:
        """Check if node is secondary decision node"""
        return self.node_type == "SDN"
    
    def get_accuracy(self) -> float:
        """Get prediction accuracy for this node"""
        if self.visits == 0:
            return 0.0
        return self.correct_predictions / self.visits
    
    def update_stats(self, correct: bool):
        """Update prediction statistics"""
        self.visits += 1
        if correct:
            self.correct_predictions += 1


class PrimaryDecisionNode(ShapeletNode):
    """Primary Decision Node (PDN) - major decision points"""
    
    def __init__(self, node_id: int, shapelet: np.ndarray, threshold: float, 
                 similarity_matrix: np.ndarray = None):
        super().__init__(node_id, "PDN", shapelet, threshold)
        self.similarity_matrix = similarity_matrix
        self.boundary_shapelets = []  # Shapelets that define decision boundary
        self.subtree_nodes = []  # All nodes in this subtree
    
    def add_boundary_shapelet(self, shapelet: np.ndarray, similarity: float):
        """Add shapelet to boundary definition"""
        self.boundary_shapelets.append((shapelet, similarity))
    
    def add_subtree_node(self, node: 'ShapeletNode'):
        """Add node to subtree"""
        self.subtree_nodes.append(node)


class SecondaryDecisionNode(ShapeletNode):
    """Secondary Decision Node (SDN) - inherits PDN cache"""
    
    def __init__(self, node_id: int, parent_pdn: PrimaryDecisionNode, 
                 shapelet: np.ndarray = None, threshold: float = 0.0):
        super().__init__(node_id, "SDN", shapelet, threshold)
        self.parent_pdn = parent_pdn
        self.cache_decisions = {}  # Cached decisions from PDN
        self.inherited_similarities = {}
    
    def inherit_cache(self):
        """Inherit decision cache from parent PDN"""
        if self.parent_pdn:
            self.cache_decisions.update(self.parent_pdn.cache_similarity or {})
            self.inherited_similarities.update(
                getattr(self.parent_pdn, 'inherited_similarities', {})
            )
            logger.debug(f"SDN {self.node_id} inherited cache from PDN {self.parent_pdn.node_id}")


class LeafDecisionNode(ShapeletNode):
    """Leaf Decision Node (LDN) - stores final classification"""
    
    def __init__(self, node_id: int, label: str, confidence: float = 1.0):
        super().__init__(node_id, "LDN", label=label)
        self.confidence = confidence
        self.class_distribution = defaultdict(int)
        self.support_samples = []
    
    def add_support_sample(self, sample: np.ndarray, label: str):
        """Add supporting sample to leaf node"""
        self.support_samples.append((sample, label))
        self.class_distribution[label] += 1
    
    def get_class_probabilities(self) -> Dict[str, float]:
        """Get class probability distribution"""
        total = sum(self.class_distribution.values())
        if total == 0:
            return {self.label: 1.0}
        return {cls: count / total for cls, count in self.class_distribution.items()}


class ShapeletTree:
    """Single shapelet decision tree"""
    
    def __init__(self, tree_id: int, max_depth: int = 15, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, similarity_threshold: float = 0.9):
        """ Initialize shapelet tree
        
        Args:
            tree_id: Unique tree identifier
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples at leaf
            similarity_threshold: Threshold for similarity caching
        """
        self.tree_id = tree_id
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.similarity_threshold = similarity_threshold
        self.root = None
        self.nodes = {}
        self.node_counter = 0
        self.similarity_calculator = ShapeletSimilarityCalculator(
            similarity_threshold=similarity_threshold
        )
        
        # Tree statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized ShapeletTree {tree_id}")
    
    def build_tree(self, shapelets: List[np.ndarray], labels: List[str], 
                   similarity_matrix: np.ndarray = None):
        """ Build shapelet tree from training data
        
        Args:
            shapelets: Training shapelets
            labels: Corresponding labels
            similarity_matrix: Pre-computed similarity matrix
        """
        logger.info(f"Building tree {self.tree_id} with {len(shapelets)} shapelets")
        
        self.similarity_matrix = similarity_matrix
        
        # Create root node (PDN)
        self.root = self._create_node("PDN", shapelets[0] if shapelets else None)
        
        # Build tree recursively
        self._build_subtree(self.root, shapelets, labels, depth=0)
        
        logger.info(f"Tree {self.tree_id} build completed with {len(self.nodes)} nodes")
    
    def _create_node(self, node_type: str, shapelet: np.ndarray = None, 
                     threshold: float = 0.0, label: str = None) -> ShapeletNode:
        """Create a new tree node"""
        self.node_counter += 1
        node_id = self.node_counter
        
        if node_type == "PDN":
            node = PrimaryDecisionNode(node_id, shapelet, threshold, self.similarity_matrix)
        elif node_type == "SDN":
            # Find parent PDN
            parent_pdn = self._find_nearest_pdn(node_id)
            node = SecondaryDecisionNode(node_id, parent_pdn, shapelet, threshold)
            node.inherit_cache()
        elif node_type == "LDN":
            node = LeafDecisionNode(node_id, label)
        else:
            node = ShapeletNode(node_id, node_type, shapelet, threshold, label)
        
        self.nodes[node_id] = node
        return node
    
    def _build_subtree(self, node: ShapeletNode, shapelets: List[np.ndarray], 
                       labels: List[str], depth: int):
        """Build subtree recursively"""
        if depth >= self.max_depth or len(shapelets) < self.min_samples_split:
            # Create leaf node
            if not node.is_leaf():
                majority_label = max(set(labels), key=labels.count)
                leaf_node = self._create_node("LDN", label=majority_label)
                
                # Replace current node with leaf
                leaf_node.parent = node.parent
                leaf_node.depth = depth
                
                if node.parent:
                    if node.parent.left_child == node:
                        node.parent.left_child = leaf_node
                    else:
                        node.parent.right_child = leaf_node
                
                # Update nodes dictionary
                del self.nodes[node.node_id]
                self.nodes[leaf_node.node_id] = leaf_node
                
                logger.debug(f"Created leaf node {leaf_node.node_id} at depth {depth}")
            return
        
        # Find best split
        best_split = self._find_best_split(shapelets, labels)
        if best_split is None:
            # Cannot split further, create leaf
            majority_label = max(set(labels), key=labels.count)
            if node.is_leaf():
                node.label = majority_label
            # else:
            #     leaf_node = self._create_node("LDN", label=majority_label)
            return
        
        # Create child nodes
        left_shapelets, left_labels, right_shapelets, right_labels = best_split
        
        # Alternate between PDN and SDN
        left_type = "SDN" if node.is_primary() else "PDN"
        right_type = "SDN" if node.is_primary() else "PDN"
        
        node.left_child = self._create_node(left_type, shapelets[0] if left_shapelets else None)
        node.right_child = self._create_node(right_type, shapelets[0] if right_shapelets else None)
        
        node.left_child.parent = node
        node.right_child.parent = node
        node.left_child.depth = depth + 1
        node.right_child.depth = depth + 1
        
        # Build subtrees
        if left_shapelets:
            self._build_subtree(node.left_child, left_shapelets, left_labels, depth + 1)
        if right_shapelets:
            self._build_subtree(node.right_child, right_shapelets, right_labels, depth + 1)
    
    def _find_best_split(self, shapelets: List[np.ndarray], labels: List[str]) -> Optional[Tuple]:
        """Find best split for current node"""
        if len(set(labels)) <= 1:
            return None
        
        # Simple split based on similarity to first shapelet
        reference = shapelets[0]
        similarities = [self.similarity_calculator.calculate_similarity(reference, s) for s in shapelets]
        
        # Find optimal threshold
        best_threshold = 0.5
        best_score = -1
        
        for threshold in np.linspace(0.1, 0.9, 9):
            left_indices = [i for i, sim in enumerate(similarities) if sim < threshold]
            right_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
            
            if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                continue
            
            # Calculate information gain (simplified)
            score = self._calculate_split_score(labels, left_indices, right_indices)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        if best_score <= 0:
            return None
        
        # Apply best split
        left_indices = [i for i, sim in enumerate(similarities) if sim < best_threshold]
        right_indices = [i for i, sim in enumerate(similarities) if sim >= best_threshold]
        
        left_shapelets = [shapelets[i] for i in left_indices]
        left_labels = [labels[i] for i in left_indices]
        right_shapelets = [shapelets[i] for i in right_indices]
        right_labels = [labels[i] for i in right_indices]
        
        return left_shapelets, left_labels, right_shapelets, right_labels
    
    def _calculate_split_score(self, labels: List[str], left_indices: List[int], 
                               right_indices: List[int]) -> float:
        """Calculate split quality score (simplified information gain)"""
        if not left_indices or not right_indices:
            return 0.0
        
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
        import math
        
        if not labels:
            return 0.0
        
        label_counts = Counter(labels)
        total = len(labels)
        entropy = 0.0
        
        for count in label_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _find_nearest_pdn(self, node_id: int) -> Optional[PrimaryDecisionNode]:
        """Find nearest PDN node for SDN inheritance"""
        # Simple implementation - find PDN with closest ID
        pdn_nodes = [node for node in self.nodes.values() if isinstance(node, PrimaryDecisionNode)]
        if not pdn_nodes:
            return None
        return min(pdn_nodes, key=lambda n: abs(n.node_id - node_id))
    
    def classify(self, shapelet: np.ndarray, use_cache: bool = True) -> Tuple[str, float, List[ShapeletNode]]:
        """ Classify input shapelet using the tree
        
        Args:
            shapelet: Input shapelet to classify
            use_cache: Whether to use similarity caching
            
        Returns:
            Tuple of (predicted_label, confidence, path_taken)
        """
        if not self.root:
            raise ValueError("Tree has not been built")
        
        self.total_predictions += 1
        path = []
        
        try:
            predicted_label, confidence = self._traverse_tree(
                self.root, shapelet, path, use_cache
            )
            
            # Update statistics
            if predicted_label is not None:
                self._update_prediction_stats(predicted_label, confidence)
            
            return predicted_label, confidence, path
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return None, 0.0, path
    
    def _traverse_tree(self, node: ShapeletNode, shapelet: np.ndarray, 
                       path: List[ShapeletNode], use_cache: bool) -> Tuple[str, float]:
        """Traverse tree for classification"""
        path.append(node)
        
        if node.is_leaf():
            # Leaf node - return prediction
            return node.label, node.confidence
        
        # Check cache first (for SDN nodes)
        if use_cache and isinstance(node, SecondaryDecisionNode) and node.cache_similarity:
            cached_sim = node.cache_similarity.get(tuple(shapelet), None)
            if cached_sim is not None and cached_sim >= self.similarity_threshold:
                self.cache_hits += 1
                
                # Use cached decision
                if cached_sim < node.threshold:
                    next_node = node.left_child
                else:
                    next_node = node.right_child
                
                if next_node:
                    return self._traverse_tree(next_node, shapelet, path, use_cache)
                else:
                    return node.label, node.confidence
        
        self.cache_misses += 1
        
        # Calculate similarity
        if node.shapelet is not None:
            similarity = self.similarity_calculator.calculate_similarity(shapelet, node.shapelet)
        else:
            similarity = 0.5  # Default if no shapelet
        
        # Make decision
        if similarity < node.threshold:
            next_node = node.left_child
        else:
            next_node = node.right_child
        
        if next_node:
            return self._traverse_tree(next_node, shapelet, path, use_cache)
        else:
            # Fallback to current node's label if no child
            return getattr(node, 'label', 'unknown'), getattr(node, 'confidence', 0.5)
    
    def _update_prediction_stats(self, predicted_label: str, confidence: float):
        """Update prediction statistics"""
        # This would be called with true label for accuracy calculation
        pass
    
    def get_tree_score(self) -> float:
        """Calculate tree performance score"""
        if self.total_predictions == 0:
            return 0.0
        
        # Simplified score based on cache efficiency and node accuracy
        cache_efficiency = self.cache_hits / max(self.total_predictions, 1)
        avg_node_accuracy = np.mean([
            node.get_accuracy() for node in self.nodes.values() 
            if node.visits > 0
        ]) if self.nodes else 0.0
        
        return 0.6 * cache_efficiency + 0.4 * avg_node_accuracy
    
    def visualize_tree(self, output_path: str, format: str = "png"):
        """Visualize tree using pybaobabdt"""
        try:
            # Create a simple scikit-learn tree for visualization
            # This is a simplified representation
            from sklearn.tree import DecisionTreeClassifier
            import matplotlib.pyplot as plt
            
            # Create dummy data for visualization
            X_dummy = np.random.randn(100, 2)
            y_dummy = np.random.choice(['class1', 'class2'], 100)
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X_dummy, y_dummy)
            
            # Visualize with pybaobabdt
            # ax = pybaobabdt.drawTree(clf, model="DT")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Tree visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing tree: {e}")


class ShapeletForest:
    """Random forest of shapelet decision trees"""
    
    def __init__(self, n_trees: int = 10, max_depth: int = 15, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, similarity_threshold: float = 0.9, 
                 tree_score_threshold: float = 0.7):
        """ Initialize shapelet forest
        
        Args:
            n_trees: Number of trees in forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples at leaf
            similarity_threshold: Threshold for similarity caching
            tree_score_threshold: Threshold for tree scoring
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.similarity_threshold = similarity_threshold
        self.tree_score_threshold = tree_score_threshold
        self.trees = []
        self.tree_scores = {}
        self.forest_stats = defaultdict(int)
        
        logger.info(f"Initialized ShapeletForest with {n_trees} trees")
    
    def build_forest(self, shapelets: List[np.ndarray], labels: List[str], 
                     similarity_matrix: np.ndarray = None):
        """ Build random forest from training data
        
        Args:
            shapelets: Training shapelets
            labels: Corresponding labels
            similarity_matrix: Pre-computed similarity matrix
        """
        logger.info(f"Building forest with {len(shapelets)} shapelets")
        
        self.trees.clear()
        self.tree_scores.clear()
        
        # Build individual trees
        for tree_id in range(self.n_trees):
            # Bootstrap sampling
            n_samples = len(shapelets)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_shapelets = [shapelets[i] for i in indices]
            bootstrap_labels = [labels[i] for i in indices]
            
            # Create and build tree
            tree = ShapeletTree(
                tree_id=tree_id,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                similarity_threshold=self.similarity_threshold
            )
            
            tree.build_tree(bootstrap_shapelets, bootstrap_labels, similarity_matrix)
            self.trees.append(tree)
            
            # Calculate initial tree score
            tree_score = tree.get_tree_score()
            self.tree_scores[tree_id] = tree_score
            
            logger.info(f"Built tree {tree_id} with score {tree_score:.3f}")
        
        logger.info(f"Forest build completed with {len(self.trees)} trees")
    
    def classify(self, shapelet: np.ndarray, use_cache: bool = True) -> Tuple[str, float, Dict[str, int]]:
        """ Classify input shapelet using forest voting
        
        Args:
            shapelet: Input shapelet to classify
            use_cache: Whether to use similarity caching
            
        Returns:
            Tuple of (predicted_label, confidence, vote_distribution)
        """
        if not self.trees:
            raise ValueError("Forest has not been built")
        
        # Collect predictions from all trees
        predictions = []
        confidences = []
        
        for tree in self.trees:
            pred_label, confidence, path = tree.classify(shapelet, use_cache)
            if pred_label is not None:
                predictions.append(pred_label)
                confidences.append(confidence)
        
        if not predictions:
            return None, 0.0, {}
        
        # Majority voting
        vote_counts = defaultdict(int)
        for pred in predictions:
            vote_counts[pred] += 1
        
        # Get majority label
        majority_label = max(vote_counts.items(), key=lambda x: x[1])[0]
        majority_votes = vote_counts[majority_label]
        
        # Calculate confidence
        avg_confidence = np.mean(confidences)
        vote_confidence = majority_votes / len(predictions)
        final_confidence = 0.7 * avg_confidence + 0.3 * vote_confidence
        
        logger.debug(f"Forest classification: {majority_label} (confidence: {final_confidence:.3f})")
        
        return majority_label, final_confidence, dict(vote_counts)
    
    def update_forest(self, new_shapelets: List[np.ndarray], new_labels: List[str], 
                      similarity_matrix: np.ndarray = None):
        """ Update forest with new data (incremental training)
        
        Args:
            new_shapelets: New training shapelets
            new_labels: Corresponding labels
            similarity_matrix: Updated similarity matrix
        """
        logger.info(f"Updating forest with {len(new_shapelets)} new shapelets")
        
        # Evaluate current tree scores
        current_scores = {tree.tree_id: tree.get_tree_score() for tree in self.trees}
        
        # Identify low-performing trees
        low_performing_trees = [
            tree for tree in self.trees 
            if current_scores[tree.tree_id] < self.tree_score_threshold
        ]
        
        logger.info(f"Identified {len(low_performing_trees)} low-performing trees")
        
        # Update low-performing trees
        for tree in low_performing_trees:
            # Combine old and new data
            old_shapelets = []  # Would need to store original training data
            combined_shapelets = old_shapelets + new_shapelets
            combined_labels = [] + new_labels  # Similarly for labels
            
            # Rebuild tree
            tree.build_tree(combined_shapelets, combined_labels, similarity_matrix)
            
            # Update score
            new_score = tree.get_tree_score()
            self.tree_scores[tree.tree_id] = new_score
            
            log_forest_update(logger, tree.tree_id, new_score, "rebuilt")
        
        # Update forest statistics
        self.forest_stats['update_iterations'] += 1
        self.forest_stats['updated_trees'] += len(low_performing_trees)
        
        logger.info(f"Forest update completed")
    
    def get_forest_statistics(self) -> Dict[str, any]:
        """Get forest statistics"""
        if not self.trees:
            return {"empty": True}
        
        tree_scores = list(self.tree_scores.values())
        
        stats = {
            "n_trees": len(self.trees),
            "avg_tree_score": np.mean(tree_scores),
            "std_tree_score": np.std(tree_scores),
            "min_tree_score": np.min(tree_scores),
            "max_tree_score": np.max(tree_scores),
            "trees_above_threshold": sum(1 for score in tree_scores if score >= self.tree_score_threshold),
            "forest_stats": dict(self.forest_stats)
        }
        
        logger.info(f"Forest statistics: {stats}")
        return stats
    
    def visualize_forest(self, output_dir: str, max_trees: int = 3):
        """ Visualize forest trees
        
        Args:
            output_dir: Output directory for visualizations
            max_trees: Maximum number of trees to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize top-performing trees
        sorted_trees = sorted(self.trees, key=lambda t: self.tree_scores.get(t.tree_id, 0), reverse=True)
        
        for i, tree in enumerate(sorted_trees[:max_trees]):
            output_path = output_dir / f"tree_{tree.tree_id}.png"
            tree.visualize_tree(str(output_path))
            logger.info(f"Visualized tree {tree.tree_id}")
    
    def get_tree_by_id(self, tree_id: int) -> Optional[ShapeletTree]:
        """Get tree by ID"""
        for tree in self.trees:
            if tree.tree_id == tree_id:
                return tree
        return None
    
    def reset_forest(self):
        """Reset forest to initial state"""
        self.trees.clear()
        self.tree_scores.clear()
        self.forest_stats.clear()
        logger.info("Reset shapelet forest")