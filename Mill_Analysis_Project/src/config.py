"""
Clean Configuration for Mill Analysis Project
Open3D-focused implementation with minimal dependencies
Professional Python 3.11 compatible
"""

from pathlib import Path
from typing import Dict, Any
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    # Get the absolute path to this file's parent directory
    return Path(__file__).parent.parent.absolute()


def get_file_paths() -> Dict[str, Path]:
    """Get standard file paths for the project."""
    root = get_project_root()
    
    return {
        'reference': root / 'data' / 'reference' / 'Mill body - vale verde - v1 (1).PLY',
        'inner_scan': root / 'data' / 'inner_scans' / '8_8-Aug-2024.ply',
        'output_dir': root / 'output',
        'intermediate_dir': root / 'output' / 'intermediate',
        'visualizations_dir': root / 'output' / 'visualizations',
        'aligned_scans_dir': root / 'output' / 'aligned_scans',
        'heatmaps_dir': root / 'output' / 'heatmaps'
    }


class ProcessingConfig:
    """Processing configuration with Open3D focus."""
    
    # File processing limits
    REFERENCE_MAX_POINTS = None  # No limit - preserve all reference points
    INNER_SCAN_MAX_POINTS = 2_000_000  # Downsample large inner scans
    
    # Noise removal parameters (V3 proven values)
    VOXEL_SIZE = 0.008
    NB_NEIGHBORS = 20
    STD_RATIO = 2.0
    DBSCAN_EPS = 0.05
    DBSCAN_MIN_POINTS = 100
    
    # Scaling parameters
    TARGET_RATIO = 1.0  # 1:1 scaling by default
    UNIT_DETECTION_THRESHOLD = 500  # Ratio threshold for unit detection
    
    # Visualization parameters
    SAMPLE_SIZE_3D = 10000  # Points for 3D visualization
    FIGURE_SIZE = (20, 12)
    DPI = 300
    
    # Heatmap generation parameters
    HEATMAP_MAX_POINTS = 100000  # Maximum points for heatmap performance
    HEATMAP_MAX_COLOR_RANGE_MM = 300.0  # Maximum range for enhanced color mapping
    HEATMAP_SAMPLE_SIZE_3D = 50000  # Points for 3D heatmap visualization
    
    # Enhanced heatmap wear bins (mm)
    ENHANCED_WEAR_BINS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 270, 300]
    
    # Enhanced heatmap colors
    ENHANCED_WEAR_COLORS = [
        '#000080', '#0066CC', '#0099FF', '#00CCFF', '#00FFCC', '#80FF80',
        '#FFFF00', '#FFB000', '#FF6000', '#FF0000', '#800000', '#660000',
        '#440000', '#220000'
    ]


class FileConfig:
    """File handling configuration."""
    
    SUPPORTED_FORMATS = ['.ply', '.pcd', '.xyz']
    DEFAULT_OUTPUT_FORMAT = '.ply'
    
    # Create output directories
    @staticmethod
    def ensure_output_dirs():
        """Ensure all output directories exist."""
        paths = get_file_paths()
        for key in ['output_dir', 'intermediate_dir', 'visualizations_dir', 'aligned_scans_dir', 'heatmaps_dir']:
            paths[key].mkdir(parents=True, exist_ok=True)


# Initialize directories on import
FileConfig.ensure_output_dirs()

# Test function for when this module is run directly
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CONFIGURATION MODULE")
    print("=" * 60)
    
    # Test project root detection
    root = get_project_root()
    print(f"Project root: {root}")
    print(f"Root exists: {root.exists()}")
    
    # Test file paths
    paths = get_file_paths()
    print(f"\nConfigured file paths:")
    for name, path in paths.items():
        exists = path.exists()
        status = "EXISTS" if exists else "MISSING"
        print(f"  {name}: {status}")
        print(f"    {path}")
    
    # Test configuration
    config = ProcessingConfig()
    print(f"\nProcessing configuration:")
    print(f"  Max inner scan points: {config.INNER_SCAN_MAX_POINTS:,}")
    print(f"  Voxel size: {config.VOXEL_SIZE}")
    print(f"  Target ratio: {config.TARGET_RATIO}")
    print(f"  Heatmap max points: {config.HEATMAP_MAX_POINTS:,}")
    print(f"  Enhanced wear bins: {len(config.ENHANCED_WEAR_BINS)} bins")
    
    print("\nConfiguration module test completed successfully")