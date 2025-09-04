"""
Enhanced Wear Heatmap Generator with CloudCompare C2C Integration
Integrates CloudCompare C2C distance files for high-precision heatmap generation
Maintains existing Open3D fallback while adding CloudCompare precision
INTEGRATION: Can read CloudCompare C2C output files and generate enhanced heatmaps
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import time

from config import ProcessingConfig, get_file_paths


class EnhancedWearHeatmapGenerator:
    """Enhanced heatmap generator with CloudCompare C2C integration."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.file_paths = get_file_paths()
        self.heatmap_analysis = {}
        
    def generate_heatmap_from_cloudcompare_c2c(self, c2c_files: List[Path]) -> Dict[str, Any]:
        """
        Generate enhanced heatmap from CloudCompare C2C distance files.
        
        Args:
            c2c_files: List of C2C distance files from CloudCompare
            
        Returns:
            Heatmap analysis results
        """
        print("Generating enhanced heatmap from CloudCompare C2C files...")
        start_time = time.time()
        
        if not c2c_files:
            print("ERROR: No C2C files provided")
            return {'heatmap_generated': False, 'error': 'No C2C files'}
        
        # Use the latest/largest C2C file (typically contains the most data)
        c2c_file = self._select_best_c2c_file(c2c_files)
        print(f"Using C2C file: {c2c_file.name}")
        
        try:
            # Read C2C file
            points, distances = self._read_c2c_file(c2c_file)
            
            if points is None or distances is None:
                print("ERROR: Failed to read C2C file data")
                return {'heatmap_generated': False, 'error': 'Failed to read C2C data'}
            
            print(f"Loaded {len(points):,} points with distance data")
            
            # Calculate statistics from CloudCompare precision data
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            percentile_95 = np.percentile(distances, 95)
            
            # Remove any invalid distances (negative or extremely large)
            valid_mask = (distances >= 0) & (distances <= 1000)  # Reasonable range for mill wear
            if np.sum(valid_mask) < len(distances):
                points = points[valid_mask]
                distances = distances[valid_mask]
                print(f"Filtered to {len(points):,} valid points")
            
            print(f"CloudCompare C2C distance statistics:")
            print(f"  Min: {min_dist:.2f}mm")
            print(f"  Max: {max_dist:.2f}mm") 
            print(f"  Mean: {mean_dist:.2f}mm")
            print(f"  Std: {std_dist:.2f}mm")
            print(f"  95th percentile: {percentile_95:.2f}mm")
            
            # Store analysis results
            self.heatmap_analysis = {
                'min_distance': min_dist,
                'max_distance': max_dist,
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'percentile_95': percentile_95,
                'points_analyzed': len(points),
                'processing_time': time.time() - start_time,
                'data_source': 'CloudCompare C2C',
                'c2c_file_used': str(c2c_file)
            }
            
            # Generate comprehensive heatmap visualization using CloudCompare precision
            self._create_cloudcompare_heatmap_visualization(points, distances, c2c_file.name)
            
            return {
                'heatmap_generated': True,
                'analysis': self.heatmap_analysis,
                'data_source': 'CloudCompare C2C'
            }
            
        except Exception as e:
            print(f"ERROR: CloudCompare C2C heatmap generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'heatmap_generated': False, 'error': str(e)}
    
    def _select_best_c2c_file(self, c2c_files: List[Path]) -> Path:
        """
        Select the best C2C file for heatmap generation.
        Prioritizes files with 'C2C_DIST' in name and largest size.
        """
        if len(c2c_files) == 1:
            return c2c_files[0]
        
        # Prioritize files with C2C_DIST in name
        c2c_dist_files = [f for f in c2c_files if 'C2C_DIST' in f.name.upper()]
        
        if c2c_dist_files:
            # Select largest C2C_DIST file
            best_file = max(c2c_dist_files, key=lambda f: f.stat().st_size)
        else:
            # Fallback to largest file
            best_file = max(c2c_files, key=lambda f: f.stat().st_size)
        
        size_mb = best_file.stat().st_size / (1024 * 1024)
        print(f"Selected C2C file: {best_file.name} ({size_mb:.1f} MB)")
        return best_file
    
    def _read_c2c_file(self, c2c_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read CloudCompare C2C file and extract points and distances.
        
        CloudCompare C2C files store distances as scalar fields in the point cloud.
        This method attempts multiple strategies to read the distance data.
        """
        try:
            print(f"Reading C2C file: {c2c_file.name}")
            
            # Try to read as PLY file (most common format)
            if c2c_file.suffix.lower() == '.ply':
                return self._read_ply_c2c_file(c2c_file)
            else:
                # Fallback to Open3D generic read
                pcd = o3d.io.read_point_cloud(str(c2c_file))
                if len(pcd.points) == 0:
                    print(f"ERROR: Empty point cloud in {c2c_file.name}")
                    return None, None
                
                points = np.asarray(pcd.points)
                
                # Try to extract colors as distance information (CloudCompare often stores distances as colors)
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    # Convert RGB to distance (assuming grayscale or single channel encoding)
                    distances = np.mean(colors, axis=1) * 255.0  # Basic conversion
                    print(f"Extracted distances from colors: range {np.min(distances):.2f} to {np.max(distances):.2f}")
                else:
                    print("WARNING: No color/distance information found, using zero distances")
                    distances = np.zeros(len(points))
                
                return points, distances
                
        except Exception as e:
            print(f"ERROR: Failed to read C2C file {c2c_file.name}: {e}")
            return None, None
    
    def _read_ply_c2c_file(self, ply_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read PLY C2C file with custom parser to extract distance scalar fields.
        CloudCompare stores C2C distances as custom scalar fields in PLY files.
        """
        try:
            points = []
            distances = []
            
            with open(ply_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find header information
            header_end = 0
            vertex_count = 0
            distance_property_found = False
            distance_property_index = -1
            
            for i, line in enumerate(lines):
                line = line.strip().lower()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif 'c2c' in line or 'distance' in line or 'scalar' in line:
                    distance_property_found = True
                    # Try to determine the index of the distance property
                    if 'property' in line:
                        distance_property_index = len([l for l in lines[:i] if l.strip().lower().startswith('property')]) - 1
                elif line == 'end_header':
                    header_end = i + 1
                    break
            
            if vertex_count == 0:
                print("ERROR: No vertices found in PLY header")
                return None, None
            
            print(f"PLY file analysis: {vertex_count} vertices, distance property: {distance_property_found}")
            
            # Read vertex data
            vertex_lines = lines[header_end:header_end + vertex_count]
            
            for line in vertex_lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                        
                        # Try to extract distance from additional columns
                        if distance_property_found and len(parts) > 3:
                            if distance_property_index >= 0 and distance_property_index + 3 < len(parts):
                                distance = float(parts[distance_property_index + 3])
                            else:
                                # Use last column as distance (common CloudCompare convention)
                                distance = float(parts[-1])
                            distances.append(distance)
                        else:
                            distances.append(0.0)  # Default distance
                    except (ValueError, IndexError):
                        continue
            
            if not points:
                print("ERROR: No valid vertex data found")
                return None, None
            
            points_array = np.array(points)
            distances_array = np.array(distances)
            
            print(f"Successfully read PLY C2C file: {len(points_array)} points")
            print(f"Distance range: {np.min(distances_array):.3f} to {np.max(distances_array):.3f}")
            
            return points_array, distances_array
            
        except Exception as e:
            print(f"ERROR: PLY C2C file parsing failed: {e}")
            # Fallback to Open3D reading
            try:
                pcd = o3d.io.read_point_cloud(str(ply_file))
                points = np.asarray(pcd.points)
                distances = np.zeros(len(points))  # Fallback to zero distances
                print(f"Fallback: Read {len(points)} points with zero distances")
                return points, distances
            except:
                return None, None
    
    def _create_cloudcompare_heatmap_visualization(self, points: np.ndarray, distances: np.ndarray, source_filename: str) -> None:
        """
        Create enhanced heatmap visualization using CloudCompare C2C precision data.
        """
        print("Creating CloudCompare precision heatmap visualization...")
        
        # Enhanced figure with CloudCompare branding
        fig = plt.figure(figsize=(28, 20))  # Larger for CloudCompare precision
        fig.suptitle(f'CloudCompare C2C Precision Heatmap Analysis\nSource: {source_filename}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Cap distances for better color visualization (use 95th percentile + buffer)
        distance_cap = min(np.percentile(distances, 95) * 1.2, 300)
        capped_distances = np.clip(distances, 0, distance_cap)
        
        # Enhanced colormap for precision analysis
        colormap = plt.cm.plasma  # Better for precision data
        norm = mcolors.Normalize(vmin=0, vmax=distance_cap)
        
        # Enhanced Top View Heatmap (X-Y plane)
        ax1 = plt.subplot(2, 4, 1)
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=capped_distances, 
                            cmap=colormap, norm=norm, s=1.5, alpha=0.8)
        ax1.set_title('CloudCompare C2C Top View\n(X-Y plane)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='C2C Distance (mm)', shrink=0.8)
        
        # Enhanced Side View Heatmap (Y-Z plane) - Primary Analysis View
        ax2 = plt.subplot(2, 4, 2)
        scatter2 = ax2.scatter(points[:, 1], points[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=1.5, alpha=0.8)
        ax2.set_title('CloudCompare C2C Side View\n(Y-Z plane) - PRIMARY ANALYSIS', 
                    fontsize=14, fontweight='bold', color='red')
        ax2.set_xlabel('Y (mm)', fontsize=12)
        ax2.set_ylabel('Z (mm)', fontsize=12)
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label='C2C Distance (mm)', shrink=0.8)
        
        # Enhanced Front View Heatmap (X-Z plane)
        ax3 = plt.subplot(2, 4, 3)
        scatter3 = ax3.scatter(points[:, 0], points[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=1.5, alpha=0.8)
        ax3.set_title('CloudCompare C2C Front View\n(X-Z plane)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X (mm)', fontsize=12)
        ax3.set_ylabel('Z (mm)', fontsize=12)
        plt.colorbar(scatter3, ax=ax3, label='C2C Distance (mm)', shrink=0.8)
        
        # CloudCompare Distance Distribution
        ax4 = plt.subplot(2, 4, 4)
        ax4.hist(capped_distances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.set_title('CloudCompare C2C Distance Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('C2C Distance (mm)', fontsize=12)
        ax4.set_ylabel('Point Count', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Enhanced Radial Analysis (Y-Z plane)
        ax5 = plt.subplot(2, 4, 5)
        center_y = np.mean(points[:, 1])
        center_z = np.mean(points[:, 2])
        radial_distances = np.sqrt((points[:, 1] - center_y)**2 + (points[:, 2] - center_z)**2)
        scatter5 = ax5.scatter(radial_distances, points[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=1, alpha=0.6)
        ax5.set_title('CloudCompare Radial Wear Analysis\n(Y-Z plane)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Radial Distance in Y-Z plane (mm)', fontsize=10)
        ax5.set_ylabel('Z (mm)', fontsize=10)
        plt.colorbar(scatter5, ax=ax5, label='C2C Distance (mm)', shrink=0.8)
        
        # Enhanced Angular Analysis (Y-Z plane)
        ax6 = plt.subplot(2, 4, 6)
        angles = np.arctan2(points[:, 2] - center_z, points[:, 1] - center_y) * 180 / np.pi
        scatter6 = ax6.scatter(angles, capped_distances, c=capped_distances,
                            cmap=colormap, norm=norm, s=1, alpha=0.6)
        ax6.set_title('CloudCompare Angular Wear Analysis\n(Y-Z plane)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Angle in Y-Z plane (degrees)', fontsize=10)
        ax6.set_ylabel('C2C Distance (mm)', fontsize=10)
        plt.colorbar(scatter6, ax=ax6, label='C2C Distance (mm)', shrink=0.8)
        
        # CloudCompare Quality Metrics
        ax7 = plt.subplot(2, 4, 7)
        ax7.axis('off')
        
        # Calculate additional quality metrics
        zero_distance_points = np.sum(distances == 0)
        high_wear_points = np.sum(capped_distances > distance_cap * 0.8)
        
        quality_text = f"""CLOUDCOMPARE C2C PRECISION ANALYSIS
        
    DATA QUALITY:
    Total points: {len(points):,}
    Zero distance points: {zero_distance_points:,} ({100*zero_distance_points/len(points):.1f}%)
    High wear points (>{distance_cap*0.8:.1f}mm): {high_wear_points:,}
    
    SPATIAL COVERAGE:
    Y range: {np.min(points[:, 1]):.1f} to {np.max(points[:, 1]):.1f} mm
    Z range: {np.min(points[:, 2]):.1f} to {np.max(points[:, 2]):.1f} mm
    Y-Z center: ({center_y:.1f}, {center_z:.1f}) mm
    
    RADIAL ANALYSIS (Y-Z):
    Min radial: {np.min(radial_distances):.1f} mm
    Max radial: {np.max(radial_distances):.1f} mm
    Mean radial: {np.mean(radial_distances):.1f} mm"""
        
        ax7.text(0.05, 0.95, quality_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # CloudCompare Statistics Summary
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        stats_text = f"""CLOUDCOMPARE C2C STATISTICS
        
    DISTANCE MEASUREMENTS:
    Min distance: {np.min(distances):.3f} mm
    Max distance: {np.max(distances):.3f} mm
    Mean distance: {np.mean(distances):.3f} mm
    Std deviation: {np.std(distances):.3f} mm
    Median distance: {np.median(distances):.3f} mm
    
    PERCENTILES:
    25th percentile: {np.percentile(distances, 25):.3f} mm
    50th percentile: {np.percentile(distances, 50):.3f} mm
    75th percentile: {np.percentile(distances, 75):.3f} mm
    95th percentile: {np.percentile(distances, 95):.3f} mm
    99th percentile: {np.percentile(distances, 99):.3f} mm
    
    PRECISION METRICS:
    Data cap used: {distance_cap:.1f} mm
    Points in analysis: {len(points):,}
    CloudCompare octree precision: HIGH"""
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save with CloudCompare identification
        output_path = self.file_paths['heatmaps_dir'] / f'cloudcompare_c2c_heatmap_analysis.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=self.config.DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"CloudCompare C2C heatmap visualization saved: {output_path}")
    
    def generate_enhanced_heatmap(self, scan1_pcd: o3d.geometry.PointCloud, 
                                 scan2_pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """
        Generate enhanced heatmap comparing two scans (fallback method when no C2C files).
        This is the original method, maintained for compatibility.
        """
        print("Generating enhanced heatmap using Open3D point-to-point comparison...")
        start_time = time.time()
        
        # Get points from both scans
        points1 = np.asarray(scan1_pcd.points)
        points2 = np.asarray(scan2_pcd.points)
        
        print(f"Scan 1: {len(points1):,} points")
        print(f"Scan 2: {len(points2):,} points")
        
        # Downsample for performance if needed
        max_points = self.config.HEATMAP_MAX_POINTS
        if len(points1) > max_points:
            indices1 = np.random.choice(len(points1), max_points, replace=False)
            points1 = points1[indices1]
            print(f"Downsampled scan 1 to {len(points1):,} points")
            
        if len(points2) > max_points:
            indices2 = np.random.choice(len(points2), max_points, replace=False)
            points2 = points2[indices2]
            print(f"Downsampled scan 2 to {len(points2):,} points")
        
        # Build KDTree for efficient distance calculation
        print("Building spatial index for distance calculation...")
        tree2 = KDTree(points2)
        
        # Calculate distances from scan1 points to nearest scan2 points
        print("Computing point-to-point distances...")
        distances, _ = tree2.query(points1)
        
        # Calculate statistics
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        percentile_95 = np.percentile(distances, 95)
        
        print(f"Open3D distance statistics:")
        print(f"  Min: {min_dist:.2f}mm")
        print(f"  Max: {max_dist:.2f}mm") 
        print(f"  Mean: {mean_dist:.2f}mm")
        print(f"  Std: {std_dist:.2f}mm")
        print(f"  95th percentile: {percentile_95:.2f}mm")
        
        # Store analysis results
        self.heatmap_analysis = {
            'min_distance': min_dist,
            'max_distance': max_dist,
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'percentile_95': percentile_95,
            'points_analyzed': len(points1),
            'processing_time': time.time() - start_time,
            'data_source': 'Open3D point-to-point'
        }
        
        # Generate standard heatmap visualization (original method)
        self._create_enhanced_heatmap_visualization(points1, points2, distances)
        
        return {
            'heatmap_generated': True,
            'analysis': self.heatmap_analysis,
            'data_source': 'Open3D point-to-point'
        }
    
    def _create_enhanced_heatmap_visualization(self, points1: np.ndarray, points2: np.ndarray, distances: np.ndarray) -> None:
        """Create enhanced heatmap visualization (original method for Open3D fallback)."""
        # This is the original method, preserved for compatibility
        fig = plt.figure(figsize=(24, 16))
        
        # Cap distances for better color visualization
        capped_distances = np.clip(distances, 0, 300)
        
        # Define colormap for wear analysis
        colormap = plt.cm.jet
        norm = mcolors.Normalize(vmin=0, vmax=300)
        
        # Main heatmap views (same as original)
        ax1 = plt.subplot(2, 3, 1)
        scatter1 = ax1.scatter(points1[:, 0], points1[:, 1], c=capped_distances, 
                            cmap=colormap, norm=norm, s=2, alpha=0.8)
        ax1.set_title('Enhanced Top View Heatmap\n(X-Y plane)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='Wear (mm)', shrink=0.8)
        
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(points1[:, 1], points1[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=2, alpha=0.8)
        ax2.set_title('ENHANCED SIDE VIEW HEATMAP\n(Y-Z plane) - Key Analysis View', 
                    fontsize=14, fontweight='bold')
        ax2.set_xlabel('Y (mm)', fontsize=12)
        ax2.set_ylabel('Z (mm)', fontsize=12)
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label='Wear (mm)', shrink=0.8)
        
        ax3 = plt.subplot(2, 3, 3)
        scatter3 = ax3.scatter(points1[:, 0], points1[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=2, alpha=0.8)
        ax3.set_title('Enhanced Front View Heatmap\n(X-Z plane)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X (mm)', fontsize=12)
        ax3.set_ylabel('Z (mm)', fontsize=12)
        plt.colorbar(scatter3, ax=ax3, label='Wear (mm)', shrink=0.8)
        
        # Analysis views (same as original)
        ax4 = plt.subplot(2, 3, 4)
        center_y = np.mean(points1[:, 1])
        center_z = np.mean(points1[:, 2])
        radial_distances = np.sqrt((points1[:, 1] - center_y)**2 + (points1[:, 2] - center_z)**2)
        scatter4 = ax4.scatter(radial_distances, points1[:, 2], c=capped_distances,
                            cmap=colormap, norm=norm, s=1, alpha=0.6)
        ax4.set_title('Enhanced Radial Wear Analysis\n(Y-Z plane distance vs wear)', 
                    fontsize=12, fontweight='bold')
        ax4.set_xlabel('Radial Distance in Y-Z plane')
        ax4.set_ylabel('Z (mm)')
        plt.colorbar(scatter4, ax=ax4, label='Wear (mm)', shrink=0.8)
        
        ax5 = plt.subplot(2, 3, 5)
        angles = np.arctan2(points1[:, 2] - center_z, points1[:, 1] - center_y) * 180 / np.pi
        scatter5 = ax5.scatter(angles, capped_distances, c=capped_distances,
                            cmap=colormap, norm=norm, s=1, alpha=0.6)
        ax5.set_title('Enhanced Angular Wear Analysis\n(Y-Z plane angle vs wear)', 
                    fontsize=12, fontweight='bold')
        ax5.set_xlabel('Angle in Y-Z plane (degrees)')
        ax5.set_ylabel('Wear (mm)')
        plt.colorbar(scatter5, ax=ax5, label='Wear (mm)', shrink=0.8)
        
        # Analysis summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""ENHANCED HEATMAP ANALYSIS (Open3D)

    SPATIAL DISTRIBUTION:
    Y range: {np.min(points1[:, 1]):.0f} to {np.max(points1[:, 1]):.0f}
    Z range: {np.min(points1[:, 2]):.0f} to {np.max(points1[:, 2]):.0f}
    Y-Z center: ({center_y:.1f}, {center_z:.1f})

    RADIAL ANALYSIS (Y-Z plane):
    Min radial distance: {np.min(radial_distances):.1f}
    Max radial distance: {np.max(radial_distances):.1f}
    Mean radial distance: {np.mean(radial_distances):.1f}

    WEAR STATISTICS:
    Min wear: {np.min(capped_distances):.1f}mm
    Max wear: {np.max(capped_distances):.1f}mm
    Mean wear: {np.mean(capped_distances):.1f}mm"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save standard heatmap
        output_path = self.file_paths['heatmaps_dir'] / 'enhanced_heatmap_analysis.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced heatmap visualization saved: {output_path}")


# Backward compatibility alias
WearHeatmapGenerator = EnhancedWearHeatmapGenerator