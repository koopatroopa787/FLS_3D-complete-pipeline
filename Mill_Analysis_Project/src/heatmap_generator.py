"""
Heatmap Generation for Mill Analysis Project
Handles two-scan comparison and wear pattern analysis
Extracted and adapted from mill_analysis_v2_1.ipynb
"""

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from typing import Tuple, Dict, Any, Optional
import time

from config import ProcessingConfig


class WearHeatmapGenerator:
    """Professional heatmap generator for mill wear analysis."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        
        # Heatmap results storage
        self.standard_heatmap = None
        self.enhanced_heatmap = None
        self.wear_distances = None
        self.enhanced_wear_distances = None
        self.heatmap_analysis = {}
        
    def generate_standard_heatmap(self, scan1_pcd: o3d.geometry.PointCloud,
                                 scan2_pcd: o3d.geometry.PointCloud,
                                 max_points: int = 100000) -> Optional[o3d.geometry.PointCloud]:
        """
        Generate standard wear heatmap comparing two scans.
        
        Args:
            scan1_pcd: First scan (reference for comparison)
            scan2_pcd: Second scan (to be colored based on wear)
            max_points: Maximum points for performance optimization
            
        Returns:
            Heatmap point cloud or None if failed
        """
        print("\nGenerating standard wear heatmap...")
        
        if scan1_pcd is None or scan2_pcd is None:
            print("ERROR: Invalid point clouds provided for heatmap generation")
            return None
            
        try:
            # Get points from both scans
            scan1_points = np.asarray(scan1_pcd.points)
            scan2_points = np.asarray(scan2_pcd.points)
            
            print(f"Original data: {len(scan1_points):,} scan1 points, {len(scan2_points):,} scan2 points")
            
            # Downsample for performance if needed
            scan1_downsampled = self._downsample_points(scan1_points, max_points)
            scan2_downsampled = self._downsample_points(scan2_points, max_points)
            
            if len(scan1_downsampled) < len(scan1_points) or len(scan2_downsampled) < len(scan2_points):
                print(f"Downsampled for performance: {len(scan1_downsampled):,} scan1, {len(scan2_downsampled):,} scan2")
            
            print("Building spatial index...")
            # Build KDTree for fast nearest neighbor search
            scan1_tree = cKDTree(scan1_downsampled)
            
            print("Computing distances...")
            # Find distances from scan2 to nearest scan1 points
            distances, nearest_indices = scan1_tree.query(scan2_downsampled, k=1)
            
            print(f"Distance statistics:")
            print(f"  Min distance: {np.min(distances):.2f}mm")
            print(f"  Max distance: {np.max(distances):.2f}mm")
            print(f"  Mean distance: {np.mean(distances):.2f}mm")
            print(f"  Std distance: {np.std(distances):.2f}mm")
            
            # Create standard color map (4-stage: Blue -> Cyan -> Green -> Yellow -> Red)
            colors = self._create_standard_colors(distances)
            
            print("Creating standard heatmap point cloud...")
            # Create heatmap point cloud
            heatmap_cloud = o3d.geometry.PointCloud()
            heatmap_cloud.points = o3d.utility.Vector3dVector(scan2_downsampled)
            heatmap_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # Copy normals if available
            if scan2_pcd.has_normals():
                scan2_normals = np.asarray(scan2_pcd.normals)
                scan2_normals_downsampled = self._downsample_points(scan2_normals, max_points)
                heatmap_cloud.normals = o3d.utility.Vector3dVector(scan2_normals_downsampled)
            
            # Store results
            self.standard_heatmap = heatmap_cloud
            self.wear_distances = distances
            
            # Calculate analysis metrics
            self.heatmap_analysis['standard'] = self._calculate_heatmap_metrics(distances, 'standard')
            
            print("Standard wear heatmap generated successfully")
            return heatmap_cloud
            
        except Exception as e:
            print(f"ERROR generating standard heatmap: {str(e)}")
            return None
    
    def generate_enhanced_heatmap(self, scan1_pcd: o3d.geometry.PointCloud,
                                 scan2_pcd: o3d.geometry.PointCloud,
                                 max_points: int = 100000,
                                 max_color_range_mm: float = 300.0) -> Optional[o3d.geometry.PointCloud]:
        """
        Generate enhanced wear heatmap with 11-stage color mapping.
        
        Args:
            scan1_pcd: First scan (reference for comparison)
            scan2_pcd: Second scan (to be colored based on wear)
            max_points: Maximum points for performance optimization
            max_color_range_mm: Maximum range for color mapping
            
        Returns:
            Enhanced heatmap point cloud or None if failed
        """
        print("\nGenerating enhanced wear heatmap with 11-stage color mapping...")
        
        if scan1_pcd is None or scan2_pcd is None:
            print("ERROR: Invalid point clouds provided for enhanced heatmap generation")
            return None
            
        try:
            # Get points from both scans
            scan1_points = np.asarray(scan1_pcd.points)
            scan2_points = np.asarray(scan2_pcd.points)
            
            print(f"Enhanced analysis - Original data: {len(scan1_points):,} scan1 points, {len(scan2_points):,} scan2 points")
            
            # Downsample for performance
            scan1_downsampled = self._downsample_points(scan1_points, max_points)
            scan2_downsampled = self._downsample_points(scan2_points, max_points)
            
            if len(scan1_downsampled) < len(scan1_points) or len(scan2_downsampled) < len(scan2_points):
                print(f"Enhanced downsampling: {len(scan1_downsampled):,} scan1, {len(scan2_downsampled):,} scan2")
            
            print("Building enhanced spatial index...")
            scan1_tree = cKDTree(scan1_downsampled)
            
            print("Computing enhanced distance analysis...")
            distances, nearest_indices = scan1_tree.query(scan2_downsampled, k=1)
            
            # Convert to millimeters for enhanced analysis
            distances_mm = distances * 1000
            
            print(f"Enhanced distance statistics:")
            print(f"  Min distance: {np.min(distances_mm):.2f}mm")
            print(f"  Max distance: {np.max(distances_mm):.2f}mm")
            print(f"  Mean distance: {np.mean(distances_mm):.2f}mm")
            print(f"  Std distance: {np.std(distances_mm):.2f}mm")
            print(f"  95th percentile: {np.percentile(distances_mm, 95):.2f}mm")
            
            # Create enhanced 11-stage color mapping
            colors = self._create_enhanced_colors(distances_mm, max_color_range_mm)
            
            print("Creating enhanced heatmap point cloud...")
            enhanced_heatmap_cloud = o3d.geometry.PointCloud()
            enhanced_heatmap_cloud.points = o3d.utility.Vector3dVector(scan2_downsampled)
            enhanced_heatmap_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # Copy normals if available
            if scan2_pcd.has_normals():
                scan2_normals = np.asarray(scan2_pcd.normals)
                scan2_normals_downsampled = self._downsample_points(scan2_normals, max_points)
                enhanced_heatmap_cloud.normals = o3d.utility.Vector3dVector(scan2_normals_downsampled)
            
            # Store results
            self.enhanced_heatmap = enhanced_heatmap_cloud
            self.enhanced_wear_distances = distances_mm
            
            # Calculate enhanced analysis metrics
            self.heatmap_analysis['enhanced'] = self._calculate_heatmap_metrics(distances_mm, 'enhanced')
            
            print("Enhanced wear heatmap generated successfully")
            return enhanced_heatmap_cloud
            
        except Exception as e:
            print(f"ERROR generating enhanced heatmap: {str(e)}")
            return None
    
    def _downsample_points(self, points: np.ndarray, target_size: int) -> np.ndarray:
        """Smart downsampling of point cloud."""
        if len(points) <= target_size:
            return points
        
        step = len(points) // target_size
        indices = np.arange(0, len(points), step)[:target_size]
        return points[indices]
    
    def _create_standard_colors(self, distances: np.ndarray) -> np.ndarray:
        """Create standard 4-stage color mapping."""
        # Normalize distances to [0, 1] range
        dist_normalized = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
        # Vectorized color calculation
        colors = np.zeros((len(distances), 3))
        
        # Blue to Cyan (0-0.25)
        mask1 = dist_normalized < 0.25
        colors[mask1, 1] = dist_normalized[mask1] * 4
        colors[mask1, 2] = 1
        
        # Cyan to Green (0.25-0.5)
        mask2 = (dist_normalized >= 0.25) & (dist_normalized < 0.5)
        colors[mask2, 1] = 1
        colors[mask2, 2] = 1 - (dist_normalized[mask2] - 0.25) * 4
        
        # Green to Yellow (0.5-0.75)
        mask3 = (dist_normalized >= 0.5) & (dist_normalized < 0.75)
        colors[mask3, 0] = (dist_normalized[mask3] - 0.5) * 4
        colors[mask3, 1] = 1
        
        # Yellow to Red (0.75-1.0)
        mask4 = dist_normalized >= 0.75
        colors[mask4, 0] = 1
        colors[mask4, 1] = 1 - (dist_normalized[mask4] - 0.75) * 4
        
        return colors
    
    def _create_enhanced_colors(self, distances_mm: np.ndarray, max_range_mm: float) -> np.ndarray:
        """Create enhanced 11-stage color mapping."""
        # Define discrete wear bins (mm)
        wear_bins = self.config.ENHANCED_WEAR_BINS
        
        # Define colors matching the bins
        wear_colors = self.config.ENHANCED_WEAR_COLORS
        
        # Create colormap and norm
        enhanced_wear_cmap = LinearSegmentedColormap.from_list('enhanced_wear_map', wear_colors, N=len(wear_bins)-1)
        norm = BoundaryNorm(wear_bins, ncolors=enhanced_wear_cmap.N)
        
        # Cap distances to max range
        capped_distances = np.clip(distances_mm, wear_bins[0], wear_bins[-1])
        beyond_cap = np.sum(distances_mm > wear_bins[-1])
        
        if beyond_cap > 0:
            print(f"Note: {beyond_cap:,} points exceed {wear_bins[-1]}mm (capped for enhanced coloring)")
        
        # Apply colormap
        colors = enhanced_wear_cmap(norm(capped_distances))[:, :3]
        
        return colors
    
    def _calculate_heatmap_metrics(self, distances: np.ndarray, heatmap_type: str) -> Dict[str, Any]:
        """Calculate comprehensive heatmap analysis metrics."""
        metrics = {
            'type': heatmap_type,
            'num_points': len(distances),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(distances)),
            'percentile_95': float(np.percentile(distances, 95)),
            'percentile_75': float(np.percentile(distances, 75)),
            'percentile_25': float(np.percentile(distances, 25))
        }
        
        return metrics
    
    def get_heatmap_results(self) -> Dict[str, Any]:
        """
        Get comprehensive heatmap results.
        
        Returns:
            Dictionary with all heatmap results and analysis
        """
        results = {
            'standard_heatmap': self.standard_heatmap,
            'enhanced_heatmap': self.enhanced_heatmap,
            'wear_distances': self.wear_distances,
            'enhanced_wear_distances': self.enhanced_wear_distances,
            'analysis': self.heatmap_analysis,
            'has_standard': self.standard_heatmap is not None,
            'has_enhanced': self.enhanced_heatmap is not None
        }
        
        return results
    
    def save_heatmap_results(self, output_dir: str) -> bool:
        """
        Save heatmap results to output directory.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            True if save successful
        """
        try:
            import os
            from pathlib import Path
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # Save standard heatmap
            if self.standard_heatmap is not None:
                standard_path = output_path / 'standard_wear_heatmap.ply'
                o3d.io.write_point_cloud(str(standard_path), self.standard_heatmap)
                saved_files.append(str(standard_path))
                print(f"Standard heatmap saved: {standard_path}")
            
            # Save enhanced heatmap
            if self.enhanced_heatmap is not None:
                enhanced_path = output_path / 'enhanced_wear_heatmap.ply'
                o3d.io.write_point_cloud(str(enhanced_path), self.enhanced_heatmap)
                saved_files.append(str(enhanced_path))
                print(f"Enhanced heatmap saved: {enhanced_path}")
            
            return True
            
        except Exception as e:
            print(f"ERROR saving heatmap results: {str(e)}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING HEATMAP GENERATOR MODULE")
    print("=" * 60)
    
    generator = WearHeatmapGenerator()
    print("Heatmap generator created successfully")
    
    print("\nHeatmap Generator Module - Ready for pipeline integration")
    print("Capabilities included:")
    print("- Standard heatmap generation (4-stage color mapping)")
    print("- Enhanced heatmap generation (11-stage color mapping)")
    print("- Two-scan comparison workflow")
    print("- Professional distance analysis")
    print("- Results storage and export")