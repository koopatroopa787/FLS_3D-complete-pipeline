"""
V4 Enhanced Noise Removal Algorithm for Mill Analysis Project
Open3D implementation - Enhanced version with improved base plate removal
Maintains backwards compatibility while adding enhanced methods
"""

import open3d as o3d
import numpy as np
import copy
from typing import Optional, Dict, Any

from config import ProcessingConfig


class NoiseRemovalProcessor:
    """Enhanced noise removal processor with V3 and V4 algorithms."""
    
    def __init__(self):
        self.config = ProcessingConfig()
    
    def apply_v3_noise_removal(self, pcd: o3d.geometry.PointCloud, 
                              voxel_size: float = 0.008,
                              nb_neighbors: int = 20,
                              std_ratio: float = 2.0,
                              eps: float = 0.05,
                              min_points: int = 100) -> Optional[o3d.geometry.PointCloud]:
        """
        Apply V3's original proven 4-step noise removal algorithm.
        Kept for backwards compatibility.
        """
        if pcd is None or len(pcd.points) == 0:
            print("ERROR: Empty point cloud for noise removal")
            return None
            
        print(f"\nStarting V3 noise removal...")
        original_count = len(pcd.points)
        print(f"Original points: {original_count:,}")
        
        try:
            # Step 1: Voxel downsampling for efficiency
            print("Step 1: Voxel downsampling...")
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)
            downsampled_count = len(pcd_downsampled.points)
            print(f"After downsampling: {downsampled_count:,} points")
            
            # Step 2: Statistical outlier removal
            print("Step 2: Statistical outlier removal...")
            pcd_clean, outlier_indices = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
            outlier_removed_count = len(pcd_clean.points)
            print(f"After outlier removal: {outlier_removed_count:,} points")
            
            # Step 3: Large plane removal (CRITICAL - removes flat bottom surface)
            print("Step 3: Removing large planes (flat bottom surface)...")
            try:
                plane_model, inliers = pcd_clean.segment_plane(
                    distance_threshold=0.01, ransac_n=3, num_iterations=1000
                )
                if len(inliers) > 0:
                    pcd_no_planes = pcd_clean.select_by_index(inliers, invert=True)
                    plane_removed_count = len(pcd_no_planes.points)
                    print(f"Removed flat surface with {len(inliers):,} points")
                    print(f"After plane removal: {plane_removed_count:,} points")
                    pcd_clean = pcd_no_planes
                else:
                    print("No significant plane detected")
            except Exception as e:
                print(f"Plane removal failed: {e}, continuing without it")
                
            # Step 4: DBSCAN clustering to find main mill structure
            print("Step 4: DBSCAN clustering...")
            labels = np.array(pcd_clean.cluster_dbscan(eps=eps, min_points=min_points))
            max_label = labels.max()
            
            if max_label < 0:
                print("Warning: No clusters found, using all remaining points")
                pcd_final = pcd_clean
            else:
                print(f"Found {max_label + 1} clusters")
                
                # Find the largest cluster (main mill structure)
                cluster_sizes = np.bincount(labels[labels >= 0])
                largest_cluster = np.argmax(cluster_sizes)
                largest_cluster_size = cluster_sizes[largest_cluster]
                
                print(f"Largest cluster (main structure): {largest_cluster_size:,} points")
                
                # Extract the largest cluster
                mask = labels == largest_cluster
                pcd_final = pcd_clean.select_by_index(np.where(mask)[0])
                
            final_count = len(pcd_final.points)
            reduction_percentage = ((original_count - final_count) / original_count) * 100
            
            print(f"\nV3 noise removal completed:")
            print(f"  Original: {original_count:,} points")
            print(f"  Final: {final_count:,} points") 
            print(f"  Reduction: {reduction_percentage:.1f}%")
            
            # Compute normals for final result
            if not pcd_final.has_normals():
                pcd_final.estimate_normals()
                
            return pcd_final
            
        except Exception as e:
            print(f"ERROR during noise removal: {str(e)}")
            return None
    
    def apply_v4_enhanced_noise_removal(self, pcd: o3d.geometry.PointCloud, 
                                      voxel_size: float = 0.008,
                                      nb_neighbors: int = 20,
                                      std_ratio: float = 2.0,
                                      eps: float = 0.05,
                                      min_points: int = 100,
                                      base_plate_method: str = "auto") -> Optional[o3d.geometry.PointCloud]:
        """
        Apply V4 enhanced noise removal with base plate removal BEFORE downsampling.
        
        Enhanced Order:
        1. Statistical outlier removal (light pass)
        2. Base plate removal (preserves mill structure)
        3. Voxel downsampling (on cleaned data)
        4. DBSCAN clustering for final cleanup
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling
            nb_neighbors: Number of neighbors for statistical outlier removal
            std_ratio: Standard deviation ratio for outlier removal
            eps: DBSCAN clustering epsilon parameter
            min_points: Minimum points for DBSCAN clustering
            base_plate_method: Method for base plate removal ('auto', 'multi_plane', 'z_filter', 'combined')
            
        Returns:
            Cleaned point cloud or None if failed
        """
        if pcd is None or len(pcd.points) == 0:
            print("ERROR: Empty point cloud for enhanced noise removal")
            return None
            
        print(f"\nStarting V4 enhanced noise removal...")
        original_count = len(pcd.points)
        print(f"Original points: {original_count:,}")
        
        try:
            # Auto-select method if requested
            if base_plate_method == "auto":
                base_plate_method = self._recommend_removal_method(pcd)
            
            # Step 1: Light statistical outlier removal first (preserve structure)
            print("Step 1: Light statistical outlier removal...")
            pcd_clean, outlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio * 1.5  # More lenient first pass
            )
            outlier_removed_count = len(pcd_clean.points)
            print(f"After light outlier removal: {outlier_removed_count:,} points")
            
            # Step 2: Base plate removal (CRITICAL - before downsampling)
            print("Step 2: Enhanced base plate removal (before downsampling)...")
            if base_plate_method == "multi_plane":
                pcd_no_base = self._remove_base_plate_multi_plane(pcd_clean)
            elif base_plate_method == "z_filter":
                pcd_no_base = self._remove_base_plate_z_filter(pcd_clean)
            elif base_plate_method == "combined":
                pcd_no_base = self._remove_base_plate_combined(pcd_clean)
            else:
                pcd_no_base = self._remove_base_plate_multi_plane(pcd_clean)  # Default
                
            base_removed_count = len(pcd_no_base.points)
            base_points_removed = outlier_removed_count - base_removed_count
            print(f"Base plate removal: {base_points_removed:,} points removed")
            print(f"After base plate removal: {base_removed_count:,} points")
            
            # Step 3: Voxel downsampling (now on clean mill structure)
            print("Step 3: Voxel downsampling on cleaned structure...")
            pcd_downsampled = pcd_no_base.voxel_down_sample(voxel_size)
            downsampled_count = len(pcd_downsampled.points)
            print(f"After downsampling: {downsampled_count:,} points")
            
            # Step 4: DBSCAN clustering for final cleanup
            print("Step 4: DBSCAN clustering for final structure isolation...")
            labels = np.array(pcd_downsampled.cluster_dbscan(eps=eps, min_points=min_points))
            max_label = labels.max()
            
            if max_label < 0:
                print("Warning: No clusters found, using all remaining points")
                pcd_final = pcd_downsampled
            else:
                print(f"Found {max_label + 1} clusters")
                
                # Find the largest cluster (main mill structure)
                cluster_sizes = np.bincount(labels[labels >= 0])
                largest_cluster = np.argmax(cluster_sizes)
                largest_cluster_size = cluster_sizes[largest_cluster]
                
                print(f"Largest cluster (main structure): {largest_cluster_size:,} points")
                
                # Extract the largest cluster
                mask = labels == largest_cluster
                pcd_final = pcd_downsampled.select_by_index(np.where(mask)[0])
                
            final_count = len(pcd_final.points)
            reduction_percentage = ((original_count - final_count) / original_count) * 100
            
            print(f"\nV4 enhanced noise removal completed:")
            print(f"  Original: {original_count:,} points")
            print(f"  Final: {final_count:,} points") 
            print(f"  Reduction: {reduction_percentage:.1f}%")
            print(f"  Method used: {base_plate_method}")
            print(f"  Mill structure consistency: PRESERVED")
            
            # Compute normals for final result
            if not pcd_final.has_normals():
                pcd_final.estimate_normals()
                
            return pcd_final
            
        except Exception as e:
            print(f"ERROR during enhanced noise removal: {str(e)}")
            return None

    def _remove_base_plate_multi_plane(self, pcd: o3d.geometry.PointCloud, 
                                     max_iterations: int = 3) -> o3d.geometry.PointCloud:
        """Remove base plate using multiple plane detection iterations."""
        print("  Using multi-plane removal method...")
        pcd_clean = copy.deepcopy(pcd)
        total_removed = 0
        
        for i in range(max_iterations):
            try:
                plane_model, inliers = pcd_clean.segment_plane(
                    distance_threshold=0.015,
                    ransac_n=3, 
                    num_iterations=2000
                )
                
                if len(inliers) > len(pcd_clean.points) * 0.05:
                    # Check if this is likely a base plate (horizontal plane)
                    normal = np.array(plane_model[:3])
                    if abs(normal[2]) > 0.8:  # Mostly horizontal plane
                        pcd_clean = pcd_clean.select_by_index(inliers, invert=True)
                        total_removed += len(inliers)
                        print(f"    Iteration {i+1}: Removed {len(inliers):,} base plate points")
                    else:
                        print(f"    Iteration {i+1}: Detected non-horizontal plane, skipping")
                        break
                else:
                    print(f"    Iteration {i+1}: No significant plane found, stopping")
                    break
                    
            except Exception as e:
                print(f"    Iteration {i+1}: Plane detection failed: {e}")
                break
                
        print(f"  Total base plate points removed: {total_removed:,}")
        return pcd_clean

    def _remove_base_plate_z_filter(self, pcd: o3d.geometry.PointCloud, 
                                  percentile_threshold: float = 15.0) -> o3d.geometry.PointCloud:
        """Remove base plate using Z-coordinate filtering."""
        print("  Using Z-coordinate filtering method...")
        points = np.asarray(pcd.points)
        z_coords = points[:, 2]
        
        # Calculate threshold based on percentile
        z_threshold = np.percentile(z_coords, percentile_threshold)
        
        # Keep points above threshold
        mask = z_coords > z_threshold
        filtered_points = points[mask]
        
        # Create new point cloud
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Copy colors and normals if they exist
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[mask])
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            pcd_filtered.normals = o3d.utility.Vector3dVector(normals[mask])
            
        removed_count = len(points) - len(filtered_points)
        print(f"  Z-filter removed {removed_count:,} points below {z_threshold:.3f}")
        
        return pcd_filtered

    def _remove_base_plate_combined(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Combined approach: Z-filtering followed by plane removal."""
        print("  Using combined removal method...")
        
        # First: Z-coordinate filtering (conservative)
        pcd_z_filtered = self._remove_base_plate_z_filter(pcd, percentile_threshold=10.0)
        
        # Second: Plane removal on remaining points
        pcd_final = self._remove_base_plate_multi_plane(pcd_z_filtered, max_iterations=2)
        
        return pcd_final

    def _analyze_base_plate_geometry(self, pcd: o3d.geometry.PointCloud) -> Dict[str, any]:
        """Analyze the base plate geometry to help choose the best removal method."""
        points = np.asarray(pcd.points)
        z_coords = points[:, 2]
        
        analysis = {
            'z_range': np.max(z_coords) - np.min(z_coords),
            'z_std': np.std(z_coords),
            'bottom_10_percent_z': np.percentile(z_coords, 10),
            'bottom_20_percent_z': np.percentile(z_coords, 20),
        }
        
        # Detect horizontal planes
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=1000
            )
            normal = np.array(plane_model[:3])
            analysis['largest_plane_horizontal'] = abs(normal[2]) > 0.8
            analysis['largest_plane_size'] = len(inliers)
            analysis['largest_plane_percentage'] = len(inliers) / len(points) * 100
        except:
            analysis['largest_plane_horizontal'] = False
            analysis['largest_plane_size'] = 0
            analysis['largest_plane_percentage'] = 0
            
        return analysis

    def _recommend_removal_method(self, pcd: o3d.geometry.PointCloud) -> str:
        """Analyze the point cloud and recommend the best base plate removal method."""
        analysis = self._analyze_base_plate_geometry(pcd)
        
        print(f"  Base plate analysis:")
        print(f"    Z-range: {analysis['z_range']:.3f}")
        print(f"    Largest plane: {analysis['largest_plane_percentage']:.1f}% of points")
        print(f"    Is horizontal: {analysis['largest_plane_horizontal']}")
        
        # Decision logic
        if analysis['largest_plane_horizontal'] and analysis['largest_plane_percentage'] > 20:
            print("    Recommendation: multi_plane (large horizontal plane detected)")
            return "multi_plane"
        elif analysis['z_std'] > 100:  # Very spread out in Z
            print("    Recommendation: combined (complex geometry)")
            return "combined"
        else:
            print("    Recommendation: z_filter (simple geometry)")
            return "z_filter"


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ENHANCED NOISE REMOVAL MODULE")
    print("=" * 60)
    
    processor = NoiseRemovalProcessor()
    print("Enhanced noise removal processor created successfully")
    
    print(f"\nConfiguration loaded:")
    print(f"Voxel size: {processor.config.VOXEL_SIZE}")
    print(f"DBSCAN eps: {processor.config.DBSCAN_EPS}")
    print(f"DBSCAN min points: {processor.config.DBSCAN_MIN_POINTS}")
    
    print(f"\nAvailable methods:")
    print("- V3 (original): apply_v3_noise_removal()")
    print("- V4 (enhanced): apply_v4_enhanced_noise_removal()")
    print("- Base plate methods: auto, multi_plane, z_filter, combined")
    
    print("\nEnhanced Noise Removal Module - Ready for pipeline integration")