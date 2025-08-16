"""
V3 Noise Removal Algorithm for Mill Analysis Project
Open3D implementation - Extracted from original massive data_loader.py
Implements proven 4-step noise removal: Voxel → Statistical → Plane → DBSCAN
"""

import open3d as o3d
import numpy as np
from typing import Optional

from config import ProcessingConfig


class NoiseRemovalProcessor:
    """V3 noise removal processor using proven 4-step algorithm."""
    
    def __init__(self):
        self.config = ProcessingConfig()
    
    def apply_v3_noise_removal(self, pcd: o3d.geometry.PointCloud, 
                              voxel_size: float = 0.008,
                              nb_neighbors: int = 20,
                              std_ratio: float = 2.0,
                              eps: float = 0.05,
                              min_points: int = 100) -> Optional[o3d.geometry.PointCloud]:
        """
        Apply V3's proven 4-step noise removal algorithm with flat surface removal.
        
        Steps:
        1. Voxel downsampling for efficiency
        2. Statistical outlier removal
        3. Large plane removal (removes flat bottom surface)
        4. DBSCAN clustering to find main structure
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling
            nb_neighbors: Number of neighbors for statistical outlier removal
            std_ratio: Standard deviation ratio for outlier removal
            eps: DBSCAN clustering epsilon parameter
            min_points: Minimum points for DBSCAN clustering
            
        Returns:
            Cleaned point cloud or None if failed
        """
        if pcd is None or len(pcd.points) == 0:
            print("ERROR: Empty point cloud for noise removal")
            return None
            
        print(f"\nStarting V3 advanced noise removal...")
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


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING V3 NOISE REMOVAL MODULE")
    print("=" * 60)
    
    processor = NoiseRemovalProcessor()
    print("V3 noise removal processor created successfully")
    
    print(f"\nConfiguration loaded:")
    print(f"Voxel size: {processor.config.VOXEL_SIZE}")
    print(f"DBSCAN eps: {processor.config.DBSCAN_EPS}")
    print(f"DBSCAN min points: {processor.config.DBSCAN_MIN_POINTS}")
    
    print("\nV3 Noise Removal Module - Ready for pipeline integration")