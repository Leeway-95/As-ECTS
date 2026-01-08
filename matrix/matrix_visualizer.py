import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from datetime import datetime
import os

from utils.logger import get_logger
from configs.config import VISUALIZATION_CONFIG

logger = get_logger(__name__)


class MatrixVisualizer:
    """Visualizes similarity matrix evolution and statistics"""
    
    def __init__(self, matrix_logs_dir: str = "./matrix_logs", output_dir: str = "./visualizations"):
        """ Initialize matrix visualizer
        
        Args:
            matrix_logs_dir: Directory containing matrix snapshots
            output_dir: Output directory for visualizations
        """
        self.matrix_logs_dir = Path(matrix_logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization settings
        self.figure_size = VISUALIZATION_CONFIG["figure_size"]
        self.dpi = VISUALIZATION_CONFIG["tree_dpi"]
        self.animation_fps = VISUALIZATION_CONFIG["matrix_animation_fps"]
        self.enable_image_generation = VISUALIZATION_CONFIG.get("enable_image_generation", False)
        
        # Color schemes
        self.cmap = self._create_custom_colormap()
        
        logger.info(f"Initialized MatrixVisualizer with output_dir={output_dir}")

    def _create_custom_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for matrix visualization"""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        cmap = LinearSegmentedColormap.from_list('custom_similarity', colors, N=100)
        return cmap

    def visualize_matrix_evolution(self, snapshot_ids: Optional[List[int]] = None, save_format: str = "png") -> List[str]:
        """ Visualize matrix evolution as a series of static plots
        
        Args:
            snapshot_ids: List of snapshot IDs to visualize (None for all)
            save_format: Output format (png, pdf, svg)
            
        Returns:
            List of generated file paths
        """
        if not self.enable_image_generation:
            return []
            
        if snapshot_ids is None:
            snapshot_ids = self._get_available_snapshots()
            
        if not snapshot_ids:
            logger.warning("No snapshots available for visualization")
            return []
        
        generated_files = []
        
        for snapshot_id in snapshot_ids:
            try:
                # Load snapshot
                snapshot_data = self._load_snapshot(snapshot_id)
                if snapshot_data is None:
                    continue
                
                # Generate visualization
                output_path = self._generate_matrix_plot(snapshot_data, snapshot_id, save_format)
                generated_files.append(output_path)
                logger.info(f"Generated matrix visualization for snapshot {snapshot_id}")
                
            except Exception as e:
                logger.error(f"Error visualizing snapshot {snapshot_id}: {e}")
        
        return generated_files

    def _load_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """Load matrix snapshot data"""
        snapshot_file = self.matrix_logs_dir / f"snapshot_{snapshot_id:03d}.npz"
        
        if not snapshot_file.exists():
            logger.warning(f"Snapshot {snapshot_id} not found")
            return None
        
        try:
            data = np.load(snapshot_file, allow_pickle=True)
            metadata = data["metadata"].item()

            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            ranking_map = {}
            if "ranking_map" in data:
                ranking_map = dict(data["ranking_map"])
            elif "ranks" in data and "indices" in data:
                ranks = data["ranks"]
                indices = data["indices"]

                if len(ranks) > 0 and indices.shape[0] > 0:
                    for i, rank in enumerate(ranks):
                        if i < indices.shape[0]:
                            ranking_map[int(rank)] = (int(indices[i, 0]), int(indices[i, 1]))
            else:
                logger.warning(f"No ranking data found in snapshot {snapshot_id}")
            
            snapshot = {
                "matrix_data": data["matrix"],
                "ranking_map": ranking_map,
                "metadata": metadata,
                "timestamp": metadata["timestamp"],
                "operation": metadata["operation"],
                "snapshot_id": metadata["snapshot_id"]
            }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            return None

    def _generate_matrix_plot(self, snapshot_data: Dict[str, Any], snapshot_id: int, save_format: str) -> str:
        """Generate matrix visualization plot"""
        matrix = snapshot_data["matrix_data"]
        metadata = snapshot_data["metadata"]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle(f'Shapelet Similarity Matrix Evolution - Snapshot {snapshot_id}', fontsize=16)

        ax1 = axes[0, 0]

        valid_mask = matrix >= 0
        if np.any(valid_mask):
            valid_data = matrix[valid_mask]
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)

            masked_matrix = np.ma.masked_where(matrix < 0, matrix)

            im1 = ax1.imshow(masked_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax1.set_title(f'Matrix State (Operation: {metadata["operation"]})')
            ax1.set_xlabel('Shapelet Index')
            ax1.set_ylabel('Shapelet Index')

            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Similarity Score')

            masked_count = np.sum(matrix < 0)
            total_count = matrix.size
            ax1.text(0.02, 0.98, f'Masked: {masked_count}/{total_count} ({masked_count/total_count*100:.1f}%)',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No valid similarity data', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14, color='red')
            ax1.set_title(f'Matrix State - No Valid Data')
        
        # Matrix statistics
        ax2 = axes[0, 1]
        self._plot_matrix_statistics(matrix, ax2, metadata)
        
        # Ranking distribution
        ax3 = axes[1, 0]
        self._plot_ranking_distribution(snapshot_data["ranking_map"], matrix, ax3)
        
        # Evolution timeline
        ax4 = axes[1, 1]
        self._plot_evolution_timeline(snapshot_data, ax4)
        
        # Add metadata text
        fig.text(0.02, 0.02, f'Timestamp: {snapshot_data["timestamp"]}\n'
                              f'Shapelets: {metadata["matrix_shape"][0]}x{metadata["matrix_shape"][1]}',
                fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"matrix_evolution_{snapshot_id:03d}.{save_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=save_format)
        plt.close()
        
        return str(output_path)

    def _plot_matrix_statistics(self, matrix: np.ndarray, ax: plt.Axes, metadata: Dict[str, Any]):
        """Plot matrix statistics"""
        valid_entries = matrix[matrix >= 0]  # Exclude placeholders
        
        if len(valid_entries) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Matrix Statistics')
            return
        
        stats = {
            'Mean': np.mean(valid_entries),
            'Std': np.std(valid_entries),
            'Min': np.min(valid_entries),
            'Max': np.max(valid_entries),
            'Median': np.median(valid_entries)
        }
        
        # Create bar plot
        bars = ax.bar(stats.keys(), stats.values(), 
                     color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
        ax.set_title('Similarity Statistics')
        ax.set_ylabel('Similarity Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Add ranking info if available
        if 'ranking_map_size' in metadata:
            ax.text(0.02, 0.98, f'Ranking pairs: {metadata["ranking_map_size"]}',
                   transform=ax.transAxes, verticalalignment='top')

    def _plot_ranking_distribution(self, ranking_map: Dict, matrix: np.ndarray, ax: plt.Axes):
        """Plot ranking distribution"""
        if not ranking_map:
            ax.text(0.5, 0.5, 'No ranking data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ranking Distribution')
            return

        similarities = []
        
        try:
            for rank_idx, indices in ranking_map.items():
                if isinstance(indices, (list, tuple)) and len(indices) == 2:
                    i, j = indices
                elif hasattr(indices, 'item'):
                    continue
                elif isinstance(indices, np.ndarray) and indices.shape == (2,):
                    i, j = indices[0], indices[1]
                else:
                    continue

                if (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer)) and
                    0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1] and matrix[i, j] >= 0):
                    similarities.append(matrix[i, j])
        
        except Exception as e:
            logger.warning(f"Error processing ranking map: {e}")
        
        if not similarities:
            ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ranking Distribution')
            return
        
        # Create histogram
        ax.hist(similarities, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title('Top Ranking Similarities')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        ax.axvline(np.mean(similarities), color='red', linestyle='--',
                  label=f'Mean: {np.mean(similarities):.3f}')
        ax.legend()

    def _plot_evolution_timeline(self, snapshot_data: Dict[str, Any], ax: plt.Axes):
        """Plot HM metrics evolution timeline"""
        try:
            # This would need to be passed in or accessible from the system
            if hasattr(self, 'hm_metrics_history') and self.hm_metrics_history:
                # Plot actual HM metrics if available
                iterations = []
                hm_scores = []
                
                for dataset_name, metrics in self.hm_metrics_history.items():
                    for metric in metrics:
                        iterations.append(metric['iteration'])
                        hm_scores.append(metric['hm_score'])
                
                if iterations and hm_scores:
                    ax.plot(iterations, hm_scores, 'b-o', linewidth=2, markersize=6)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('HM Score')
                    ax.set_title('HM Metrics Evolution')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1)
                    return
        except Exception as e:
            logger.debug(f"Could not plot HM metrics: {e}")

        sample_iterations = [1, 2, 3, 4, 5]
        sample_hm_scores = [0.3, 0.45, 0.52, 0.61, 0.68]  # Simulated improving HM scores
        
        ax.plot(sample_iterations, sample_hm_scores, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('HM Score')
        ax.set_title('HM Metrics Evolution (Sample)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.9, 'HM = 2*(Accuracy*Earliness)/(Accuracy+Earliness)',
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def create_matrix_animation(self, snapshot_ids: Optional[List[int]] = None, duration: int = 10) -> str:
        """ Create animated visualization of matrix evolution
        
        Args:
            snapshot_ids: List of snapshot IDs to animate (None for all)
            duration: Animation duration in seconds
            
        Returns:
            Path to generated animation file
        """
        if not self.enable_image_generation:
            return ""
            
        if snapshot_ids is None:
            snapshot_ids = self._get_available_snapshots()
            
        if not snapshot_ids:
            logger.warning("No snapshots available for animation")
            return ""
        
        try:
            # First generate individual PNG plots for all snapshots
            logger.info("Generating individual matrix evolution plots for animation...")
            png_files = self.visualize_matrix_evolution(snapshot_ids=snapshot_ids, save_format="png")
            
            if not png_files:
                logger.warning("No PNG files generated for animation")
                return ""
            
            # Load all snapshots
            snapshots = []
            for snapshot_id in snapshot_ids:
                snapshot_data = self._load_snapshot(snapshot_id)
                if snapshot_data:
                    snapshots.append(snapshot_data)
            
            if not snapshots:
                logger.warning("No valid snapshots for animation")
                return ""
            
            # Create animation
            output_path = self.output_dir / "matrix_evolution_animation.gif"
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Shapelet Similarity Matrix Evolution Animation', fontsize=16)
            
            # Initialize plots
            ax_matrix = axes[0]
            ax_stats = axes[1]
            
            # Create initial matrix plot
            matrix_data = snapshots[0]["matrix_data"]
            im = ax_matrix.imshow(matrix_data, cmap=self.cmap, aspect='auto', vmin=0, vmax=1)
            ax_matrix.set_title(f'Matrix Evolution (Frame 1/{len(snapshots)})')
            ax_matrix.set_xlabel('Shapelet Index')
            ax_matrix.set_ylabel('Shapelet Index')
            
            # Create statistics plot
            stats_line, = ax_stats.plot([], [], 'b-', linewidth=2)
            ax_stats.set_xlim(0, len(snapshots))
            ax_stats.set_ylim(0, 1)
            ax_stats.set_title('Matrix Statistics Over Time')
            ax_stats.set_xlabel('Snapshot')
            ax_stats.set_ylabel('Mean Similarity')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_matrix, shrink=0.8)
            cbar.set_label('Similarity Score')
            
            # Animation data
            stats_data = []
            for i, snapshot in enumerate(snapshots):
                matrix = snapshot["matrix_data"]
                valid_entries = matrix[matrix >= 0]
                if len(valid_entries) > 0:
                    stats_data.append(np.mean(valid_entries))
                else:
                    stats_data.append(0.0)
            
            def animate(frame):
                # Update matrix
                matrix_data = snapshots[frame]["matrix_data"]
                im.set_array(matrix_data)
                
                # Update title
                metadata = snapshots[frame]["metadata"]
                ax_matrix.set_title(f'Matrix Evolution (Frame {frame+1}/{len(snapshots)}) - {metadata["operation"]}')
                
                # Update statistics plot
                if frame < len(stats_data):
                    x_data = list(range(frame + 1))
                    y_data = stats_data[:frame + 1]
                    stats_line.set_data(x_data, y_data)
                    
                    # Adjust y-axis if needed
                    if y_data:
                        ax_stats.set_ylim(min(y_data) * 0.9, max(y_data) * 1.1)
                
                return [im, stats_line]
            
            # Create animation
            anim = animation.FuncAnimation(
                fig, animate, frames=len(snapshots),
                interval=duration * 1000 // len(snapshots), blit=True
            )
            
            # Save animation
            anim.save(str(output_path), writer='pillow', fps=self.animation_fps, dpi=self.dpi)
            plt.close()
            
            logger.info(f"Created matrix evolution animation: {output_path}")
            
            # Clean up PNG files after animation is created
            if png_files:
                logger.info(f"Cleaning up {len(png_files)} PNG files after animation creation...")
                for png_file in png_files:
                    try:
                        if os.path.exists(png_file):
                            os.remove(png_file)
                            logger.debug(f"Deleted PNG file: {png_file}")
                    except Exception as e:
                        logger.warning(f"Could not delete PNG file {png_file}: {e}")
                
                logger.info(f"Successfully cleaned up PNG files after animation creation")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
            return ""

    def _get_top_rankings(self, snapshot: Dict[str, Any], n: int = 5) -> List[Dict[str, float]]:
        """Get top N rankings from snapshot"""
        ranking_map = snapshot["ranking_map"]
        matrix = snapshot["matrix_data"]
        
        top_rankings = []
        
        for rank_idx, (i, j) in ranking_map.items():
            if i < matrix.shape[0] and j < matrix.shape[1] and matrix[i, j] >= 0:
                top_rankings.append({
                    "rank": rank_idx,
                    "shapelet_pair": (i, j),
                    "similarity": float(matrix[i, j])
                })
        
        # Sort by similarity and take top N
        top_rankings.sort(key=lambda x: x["similarity"], reverse=True)
        return top_rankings[:n]

    def _get_available_snapshots(self) -> List[int]:
        """Get list of available snapshot IDs"""
        snapshot_files = list(self.matrix_logs_dir.glob("snapshot_*.npz"))
        snapshot_ids = []
        
        for file in snapshot_files:
            try:
                # Extract snapshot ID from filename
                id_str = file.stem.replace("snapshot_", "")
                snapshot_id = int(id_str)
                snapshot_ids.append(snapshot_id)
            except ValueError:
                continue
        
        return sorted(snapshot_ids)


# Utility functions
def create_matrix_visualizer(matrix_logs_dir: str = "./matrix_logs", output_dir: str = "./visualizations") -> MatrixVisualizer:
    """Factory function to create matrix visualizer"""
    return MatrixVisualizer(matrix_logs_dir, output_dir)

def create_matrix_animation(matrix_logs_dir: str, output_dir: str, snapshot_ids: Optional[List[int]] = None, duration: int = 10) -> str:
    """Convenience function to create matrix animation"""
    visualizer = create_matrix_visualizer(matrix_logs_dir, output_dir)
    return visualizer.create_matrix_animation(snapshot_ids, duration)