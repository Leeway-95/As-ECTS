import numpy as np
from typing import List
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


from utils.logger import get_logger, get_progress_manager
from matrix.matrix_visualizer import MatrixVisualizer

logger = get_logger(__name__)
progress_manager = get_progress_manager()


class EnhancedMatrixVisualizer:
    """Matrix visualizer with comprehensive visualization capabilities"""

    def __init__(self, matrix_logs_dir: str = "./matrix_logs", output_dir: str = "./visualizations"):
        """
        Initialize matrix visualizer

        Args:
            matrix_logs_dir: Directory for matrix snapshots
            output_dir: Output directory for visualizations
        """
        self.matrix_logs_dir = Path(matrix_logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base visualizer
        self.base_visualizer = MatrixVisualizer(
            matrix_logs_dir=matrix_logs_dir,
            output_dir=output_dir
        )

        logger.info(f"🎨 Matrix visualizer initialized")
        logger.info(f"📁 Matrix logs: {self.matrix_logs_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")

    def create_matrix_heatmap(self, similarity_matrix, dataset_name: str) -> List[str]:
        """Create matrix heatmap visualization"""
        files = []

        try:
            # Get current matrix data
            matrix_data = similarity_matrix.matrix[:similarity_matrix.shapelet_count,
                                                  :similarity_matrix.shapelet_count]

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Full matrix heatmap
            im1 = ax1.imshow(matrix_data, cmap='viridis', aspect='auto')
            ax1.set_title(f'Similarity Matrix - {dataset_name}\n(Full Matrix)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Shapelet Index')
            ax1.set_ylabel('Shapelet Index')

            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Similarity Score', rotation=270, labelpad=20)

            # Statistics heatmap (excluding diagonal and invalid values)
            valid_mask = (matrix_data >= 0) & (matrix_data != 0)
            if np.any(valid_mask):
                stats_data = matrix_data.copy()
                stats_data[~valid_mask] = np.nan

                im2 = ax2.imshow(stats_data, cmap='plasma', aspect='auto')
                ax2.set_title(f'Similarity Matrix - {dataset_name}\n(Valid Values Only)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Shapelet Index')
                ax2.set_ylabel('Shapelet Index')

                # Add colorbar
                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label('Similarity Score', rotation=270, labelpad=20)

                # Add statistics text
                valid_values = matrix_data[valid_mask]
                stats_text = f'Valid pairs: {len(valid_values)}\n'
                stats_text += f'Mean: {np.nanmean(stats_data):.3f}\n'
                stats_text += f'Std: {np.nanstd(stats_data):.3f}\n'
                stats_text += f'Min: {np.nanmin(stats_data):.3f}\n'
                stats_text += f'Max: {np.nanmax(stats_data):.3f}'

                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Save figure
            filename = self.output_dir / f"matrix_heatmap_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            files.append(str(filename))

            plt.close()

            logger.info(f"✅ Matrix heatmap saved to {filename}")

        except Exception as e:
            logger.error(f"❌ Error creating matrix heatmap: {e}")

        return files

    def create_evolution_timeline(self, similarity_matrix, dataset_name: str) -> List[str]:
        """Create matrix evolution timeline visualization"""
        files = []

        try:
            evolution_history = similarity_matrix.get_evolution_history()

            if not evolution_history:
                logger.warning(f"⚠️ No evolution history available for {dataset_name}")
                return files

            # Extract data from history
            timestamps = [entry['timestamp'] for entry in evolution_history]
            shapelet_counts = [entry['metadata'].get('shapelets', 0) for entry in evolution_history]
            matrix_sizes = [entry['metadata'].get('matrix_size', 0) for entry in evolution_history]

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Plot both metrics
            ax2 = ax.twinx()

            line1 = ax.plot(range(len(timestamps)), shapelet_counts, 'b-o', linewidth=2, markersize=6, label='Shapelets')
            line2 = ax2.plot(range(len(timestamps)), matrix_sizes, 'r-s', linewidth=2, markersize=6, label='Matrix Size')

            ax.set_xlabel('Snapshot Index')
            ax.set_ylabel('Number of Shapelets', color='b')
            ax2.set_ylabel('Matrix Size', color='r')

            ax.set_title(f'Matrix Evolution Timeline - {dataset_name}', fontsize=14, fontweight='bold')

            # Color the axes
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')

            # Add grid
            ax.grid(True, alpha=0.3)

            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()

            # Save figure
            filename = self.output_dir / f"evolution_timeline_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            files.append(str(filename))

            plt.close()

            logger.info(f"✅ Evolution timeline saved to {filename}")

        except Exception as e:
            logger.error(f"❌ Error creating evolution timeline: {e}")

        return files

    def create_similarity_distribution(self, similarity_matrix, dataset_name: str) -> List[str]:
        """Create similarity distribution visualization"""
        files = []

        try:
            matrix_data = similarity_matrix.matrix[:similarity_matrix.shapelet_count,
                                                  :similarity_matrix.shapelet_count]

            # Filter valid similarities (excluding diagonal and invalid values)
            valid_mask = (matrix_data >= 0) & (matrix_data != 0)
            valid_similarities = matrix_data[valid_mask]

            if len(valid_similarities) == 0:
                logger.warning(f"⚠️ No valid similarities found for {dataset_name}")
                return files

            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Histogram
            ax1.hist(valid_similarities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Similarity Distribution - {dataset_name}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Similarity Score')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

            # Box plot
            ax2.boxplot(valid_similarities, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
            ax2.set_title(f'Similarity Box Plot - {dataset_name}', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Similarity Score')
            ax2.grid(True, alpha=0.3)

            # Cumulative distribution
            sorted_similarities = np.sort(valid_similarities)
            cumulative = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)
            ax3.plot(sorted_similarities, cumulative, 'purple', linewidth=2)
            ax3.set_title(f'Cumulative Similarity Distribution - {dataset_name}', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Similarity Score')
            ax3.set_ylabel('Cumulative Probability')
            ax3.grid(True, alpha=0.3)

            # Statistics summary
            ax4.axis('off')
            stats_text = f'Statistics for {dataset_name}:\n\n'
            stats_text += f'Total valid pairs: {len(valid_similarities)}\n'
            stats_text += f'Mean: {np.mean(valid_similarities):.4f}\n'
            stats_text += f'Median: {np.median(valid_similarities):.4f}\n'
            stats_text += f'Std: {np.std(valid_similarities):.4f}\n'
            stats_text += f'Min: {np.min(valid_similarities):.4f}\n'
            stats_text += f'Max: {np.max(valid_similarities):.4f}\n'
            stats_text += f'Q1: {np.percentile(valid_similarities, 25):.4f}\n'
            stats_text += f'Q3: {np.percentile(valid_similarities, 75):.4f}'

            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.tight_layout()

            # Save figure
            filename = self.output_dir / f"similarity_distribution_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            files.append(str(filename))

            plt.close()

            logger.info(f"✅ Similarity distribution saved to {filename}")

        except Exception as e:
            logger.error(f"❌ Error creating similarity distribution: {e}")

        return files