"""
Enhanced V3 Noise Removal for Mill Analysis Project
Open3D implementation with comprehensive cleaning algorithms
PROVEN ALGORITHMS - Maintains original V3 effectiveness
Enhanced with automatic file saving to noise_removed_folder
"""

import open3d as o3d
import numpy as np
import copy
from typing import Tuple, Optional, Dict, Any
import time
from pathlib import Path

from config import ProcessingConfig, get_file_paths


class NoiseRemovalProcessor:
    """
    Enhanced V3 noise removal processor with automatic file saving.
    Maintains proven V3 algorithms while adding file persistence.
    """
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.file_paths = get_file_paths()
        
        # Ensure noise removed directory exists
        self.file_paths['noise_removed_dir'].mkdir(parents=True, exist_ok=True)
        
        print("NoiseRemovalProcessor initialized (V3 Enhanced + File Saving)")
        print(f"  Noise removed files will be saved to: {self.file_paths['noise_removed_dir']}")
        print(f"  Save format: {self.config.NOISE_REMOVED_FORMAT}")
    
    def apply_v3_noise_removal(self, pcd: o3d.geometry.PointCloud, 
                             scan_identifier: str = "unknown") -> o3d.geometry.PointCloud:
        """
        Apply V3 noise removal algorithm with IMPROVED RETENTION and automatic file saving.
        
        Args:
            pcd: Input point cloud
            scan_identifier: Identifier for the scan (used in filename)
            
        Returns:
            Cleaned point cloud
        """
        print("\n" + "=" * 50)
        print("=== V3 NOISE REMOVAL WITH IMPROVED RETENTION ===")
        print("=" * 50)
        
        start_time = time.time()
        input_points = len(pcd.points)
        print(f"Input points: {input_points:,}")
        
        # Calculate target retention for large datasets
        if input_points > 20_000_000:
            target_min = 10_000_000
            target_max = 12_000_000
        elif input_points > 10_000_000:
            target_min = int(input_points * 0.23)  # 23% minimum retention
            target_max = int(input_points * 0.30)  # 30% maximum retention
        else:
            target_min = max(int(input_points * 0.80), 1_000_000)  # 80% retention for smaller datasets
            target_max = input_points
        
        print(f"Target retention: {target_min:,}-{target_max:,} points ({target_min/input_points*100:.1f}-{target_max/input_points*100:.1f}%)")
        
        # Step 1: IMPROVED Initial Downsampling for very large datasets
        working_pcd = self._apply_improved_initial_downsampling(pcd, target_max)
        
        # Step 2: Bottom Plate Removal (V3 original method)
        working_pcd = self._apply_bottom_plate_removal(working_pcd)
        
        # Step 3: Statistical Outlier Removal (original strength)
        working_pcd = self._apply_statistical_outlier_removal(working_pcd)
        
        # Step 4: SELECTIVE V3 DBSCAN Clustering (keeps significant clusters)
        working_pcd = self._apply_selective_v3_dbscan(working_pcd)
        
        # Final statistics
        output_points = len(working_pcd.points)
        retention_rate = (output_points / input_points) * 100
        processing_time = time.time() - start_time
        
        print(f"\nCORRECTED V3 Processing Results:")
        print(f"  Input points: {input_points:,}")
        print(f"  Output points: {output_points:,}")
        print(f"  Overall retention: {retention_rate:.1f}%")
        
        if output_points < target_min:
            print(f"  BELOW TARGET: {output_points:,} points retained (target: {target_min:,}-{target_max:,})")
        elif output_points > target_max:
            print(f"  ABOVE TARGET: {output_points:,} points retained (target: {target_min:,}-{target_max:,})")
        else:
            print(f"  WITHIN TARGET: {output_points:,} points retained (target: {target_min:,}-{target_max:,})")
        
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        # SAVE CLEANED FILE AUTOMATICALLY
        cleaned_pcd = self._save_cleaned_file(working_pcd, scan_identifier)
        
        return cleaned_pcd
    
    def _apply_improved_initial_downsampling(self, pcd: o3d.geometry.PointCloud, target_max: int) -> o3d.geometry.PointCloud:
        """IMPROVED initial downsampling that preserves more detail."""
        input_count = len(pcd.points)
        
        if input_count <= target_max:
            print(f"No initial downsampling needed: {input_count:,} points")
            return copy.deepcopy(pcd)
        
        print(f"Input exceeds safe limit, applying IMPROVED initial downsampling")
        print(f"  IMPROVED initial downsampling: {input_count:,} -> targeting {target_max:,}")
        
        # More conservative voxel sizes for better preservation
        test_voxels = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015]
        
        best_pcd = None
        for voxel_size in test_voxels:
            test_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            test_count = len(test_pcd.points)
            print(f"  Testing voxel {voxel_size}: {test_count:,} points")
            
            # Accept if within reasonable range of target
            if test_count <= target_max * 1.2:  # Allow 20% over target
                best_pcd = test_pcd
                print(f"  Using voxel size {voxel_size}")
                break
        
        if best_pcd is None:
            # Fallback: use largest voxel if nothing fits
            best_pcd = pcd.voxel_down_sample(voxel_size=test_voxels[-1])
            print(f"  Fallback: using largest voxel {test_voxels[-1]}")
        
        final_count = len(best_pcd.points)
        retention = (final_count / input_count) * 100
        print(f"  Achieved: {final_count:,} points ({retention:.1f}% initial retention)")
        
        print(f"Working with: {final_count:,} points")
        return best_pcd
    
    def _apply_bottom_plate_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply V3 bottom plate removal (original method)."""
        print("Applying BOTTOM PLATE REMOVAL (V3 original method)...")
        
        input_count = len(pcd.points)
        print(f"  V3 Bottom Plate Removal - Input: {input_count:,} points")
        
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            print("  Warning: Empty point cloud")
            return pcd
        
        # Z-axis analysis for bottom plate identification
        z_coords = points[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        z_range = z_max - z_min
        
        print(f"  Z-range: {z_min:.2f} to {z_max:.2f} (range: {z_range:.2f})")
        
        # V3 method: Remove points in bottom 10% of Z range
        bottom_threshold = z_min + (z_range * 0.10)  # Bottom 10%
        print(f"  Bottom threshold: {bottom_threshold:.2f}")
        
        # Keep points above threshold
        keep_mask = z_coords > bottom_threshold
        points_filtered = points[keep_mask]
        
        # Create cleaned point cloud
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(points_filtered)
        
        # Copy colors if they exist
        if len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)
            cleaned_pcd.colors = o3d.utility.Vector3dVector(colors[keep_mask])
        
        # Copy normals if they exist
        if len(pcd.normals) > 0:
            normals = np.asarray(pcd.normals)
            cleaned_pcd.normals = o3d.utility.Vector3dVector(normals[keep_mask])
        
        output_count = len(cleaned_pcd.points)
        removed_count = input_count - output_count
        removal_rate = (removed_count / input_count) * 100
        
        print(f"  Bottom plate removal: {removed_count:,} points removed ({removal_rate:.1f}%)")
        print(f"  Remaining: {output_count:,} points")
        
        return cleaned_pcd
    
    def _apply_statistical_outlier_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply statistical outlier removal (original strength)."""
        print("Applying statistical outlier removal (original strength)...")
        
        input_count = len(pcd.points)
        
        if input_count == 0:
            return pcd
        
        # V3 settings - proven effective
        nb_neighbors = self.config.NB_NEIGHBORS  # 20
        std_ratio = self.config.STD_RATIO  # 2.0
        
        cleaned_pcd, outlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        
        output_count = len(cleaned_pcd.points)
        removed_count = len(outlier_indices)
        removal_rate = (removed_count / input_count) * 100
        
        print(f"  Statistical removal: {removed_count:,} removed ({removal_rate:.1f}%)")
        
        return cleaned_pcd
    
    def _apply_selective_v3_dbscan(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Apply SELECTIVE V3 DBSCAN that keeps significant clusters."""
        print("Applying SELECTIVE V3 DBSCAN clustering...")
        
        input_count = len(pcd.points)
        
        if input_count == 0:
            return pcd
        
        # V3 DBSCAN parameters
        eps = self.config.DBSCAN_EPS  # 0.05
        min_points = self.config.DBSCAN_MIN_POINTS  # 100
        
        print(f"  SELECTIVE V3 DBSCAN - eps: {eps}, min_points: {min_points}")
        
        # Apply DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        # Analyze clusters
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 is noise
        
        print(f"  Found {num_clusters} clusters")
        
        if num_clusters == 0:
            print("  No clusters found, returning original")
            return pcd
        
        # Find cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_sizes[label] = np.sum(labels == label)
        
        # Sort clusters by size
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_clusters:
            print("  No valid clusters found")
            return pcd
        
        largest_cluster_size = sorted_clusters[0][1]
        print(f"  Largest cluster: {largest_cluster_size:,} points")
        
        # SELECTIVE retention: Keep significant clusters (>=1% of largest cluster)
        significance_threshold = max(largest_cluster_size * 0.01, 100)  # At least 1% or 100 points
        print(f"  Significance threshold: {significance_threshold:.0f} points")
        
        # Select significant clusters
        significant_clusters = []
        total_kept_points = 0
        
        for i, (label, size) in enumerate(sorted_clusters):
            if size >= significance_threshold:
                significant_clusters.append(label)
                total_kept_points += size
                print(f"  Keeping significant cluster {label}: {size:,} points")
            else:
                print(f"  Rejecting small cluster {label}: {size:,} points (below threshold)")
        
        if not significant_clusters:
            # Fallback: keep largest cluster
            significant_clusters = [sorted_clusters[0][0]]
            total_kept_points = sorted_clusters[0][1]
            print(f"  Fallback: keeping only largest cluster")
        
        # Create mask for significant clusters
        keep_mask = np.zeros(len(labels), dtype=bool)
        for cluster_label in significant_clusters:
            keep_mask |= (labels == cluster_label)
        
        # Extract points from significant clusters
        points = np.asarray(pcd.points)
        points_kept = points[keep_mask]
        
        # Create cleaned point cloud
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(points_kept)
        
        # Copy colors if they exist
        if len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)
            cleaned_pcd.colors = o3d.utility.Vector3dVector(colors[keep_mask])
        
        # Copy normals if they exist
        if len(pcd.normals) > 0:
            normals = np.asarray(pcd.normals)
            cleaned_pcd.normals = o3d.utility.Vector3dVector(normals[keep_mask])
        
        output_count = len(cleaned_pcd.points)
        removed_count = input_count - output_count
        removal_rate = (removed_count / input_count) * 100
        retention_rate = (output_count / input_count) * 100
        
        print(f"  Total clusters kept: {len(significant_clusters)}")
        print(f"  Total points in kept clusters: {output_count:,}")
        print(f"  SELECTIVE V3 DBSCAN: Kept {len(significant_clusters)} significant clusters")
        print(f"  Retained: {output_count:,} points ({retention_rate:.1f}%)")
        print(f"  Removed: {removed_count:,} points ({removal_rate:.1f}%)")
        
        return cleaned_pcd
    
    def _save_cleaned_file(self, cleaned_pcd: o3d.geometry.PointCloud, 
                          scan_identifier: str) -> o3d.geometry.PointCloud:
        """
        Save the cleaned point cloud to noise_removed_folder.
        
        Args:
            cleaned_pcd: Cleaned point cloud to save
            scan_identifier: Identifier for naming the file
            
        Returns:
            Same cleaned point cloud (for pipeline continuity)
        """
        if not self.config.SAVE_NOISE_REMOVED_FILES:
            print("File saving disabled in configuration")
            return cleaned_pcd
        
        try:
            # Create filename from scan identifier
            clean_identifier = scan_identifier.replace(" ", "_").replace("-", "_")
            filename = f"noise_removed_{clean_identifier}{self.config.NOISE_REMOVED_FORMAT}"
            save_path = self.file_paths['noise_removed_dir'] / filename
            
            print(f"\nSaving cleaned file: {filename}")
            print(f"  Path: {save_path}")
            print(f"  Points: {len(cleaned_pcd.points):,}")
            
            # Save the file
            success = o3d.io.write_point_cloud(str(save_path), cleaned_pcd)
            
            if success:
                # Get file size
                file_size_mb = save_path.stat().st_size / (1024 * 1024)
                print(f"  File saved successfully: {file_size_mb:.1f} MB")
                print(f"SUCCESS: Noise-removed file saved to {save_path}")
            else:
                print(f"ERROR: Failed to save file to {save_path}")
        
        except Exception as e:
            print(f"ERROR: Exception while saving cleaned file: {e}")
        
        return cleaned_pcd
    
    def get_saved_file_path(self, scan_identifier: str) -> Path:
        """
        Get the path to a saved noise-removed file.
        
        Args:
            scan_identifier: Identifier used when saving the file
            
        Returns:
            Path to the saved file
        """
        clean_identifier = scan_identifier.replace(" ", "_").replace("-", "_")
        filename = f"noise_removed_{clean_identifier}{self.config.NOISE_REMOVED_FORMAT}"
        return self.file_paths['noise_removed_dir'] / filename
    
    def load_saved_cleaned_file(self, scan_identifier: str) -> Optional[o3d.geometry.PointCloud]:
        """
        Load a previously saved noise-removed file.
        
        Args:
            scan_identifier: Identifier used when saving the file
            
        Returns:
            Loaded point cloud or None if file doesn't exist
        """
        file_path = self.get_saved_file_path(scan_identifier)
        
        if not file_path.exists():
            print(f"Saved cleaned file not found: {file_path}")
            return None
        
        try:
            print(f"Loading saved cleaned file: {file_path.name}")
            cleaned_pcd = o3d.io.read_point_cloud(str(file_path))
            
            if len(cleaned_pcd.points) > 0:
                print(f"  Loaded: {len(cleaned_pcd.points):,} points")
                return cleaned_pcd
            else:
                print(f"  ERROR: Loaded file is empty")
                return None
                
        except Exception as e:
            print(f"ERROR: Failed to load saved cleaned file: {e}")
            return None
    
    def apply_v3_noise_removal_with_save(self, pcd: o3d.geometry.PointCloud, 
                                       scan_identifier: str = "unknown", 
                                       force_reprocess: bool = False) -> o3d.geometry.PointCloud:
        """
        Apply V3 noise removal with automatic saving and optional loading of existing cleaned files.
        
        Args:
            pcd: Input point cloud
            scan_identifier: Identifier for the scan
            force_reprocess: If True, always reprocess even if cleaned file exists
            
        Returns:
            Cleaned point cloud
        """
        # Check if cleaned file already exists
        if not force_reprocess and self.config.SAVE_NOISE_REMOVED_FILES:
            existing_cleaned = self.load_saved_cleaned_file(scan_identifier)
            if existing_cleaned is not None:
                print(f"Using existing cleaned file for {scan_identifier}")
                return existing_cleaned
        
        # Process and save
        print(f"Processing and saving cleaned file for {scan_identifier}")
        return self.apply_v3_noise_removal(pcd, scan_identifier)
    
    # Legacy method for compatibility
    def apply_noise_removal(self, pcd: o3d.geometry.PointCloud, 
                          scan_identifier: str = "unknown") -> o3d.geometry.PointCloud:
        """Legacy method - redirects to V3 method with saving."""
        return self.apply_v3_noise_removal_with_save(pcd, scan_identifier)


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING V3 NOISE REMOVAL MODULE - ENHANCED WITH FILE SAVING")
    print("=" * 80)
    
    try:
        # Test processor initialization
        processor = NoiseRemovalProcessor()
        
        print(f"V3 Noise removal processor created successfully")
        print(f"  Noise removed directory: {processor.file_paths['noise_removed_dir']}")
        print(f"  Save files: {processor.config.SAVE_NOISE_REMOVED_FILES}")
        print(f"  Save format: {processor.config.NOISE_REMOVED_FORMAT}")
        
        print(f"\nV3 NOISE REMOVAL PIPELINE:")
        print("1. IMPROVED Initial Downsampling (conservative voxel sizes)")
        print("2. Bottom Plate Removal (V3 original method - bottom 10%)")
        print("3. Statistical Outlier Removal (nb_neighbors=20, std_ratio=2.0)")
        print("4. SELECTIVE V3 DBSCAN (keeps significant clusters)")
        print("5. AUTOMATIC FILE SAVING to noise_removed_folder")
        
        print(f"\nFILE MANAGEMENT FEATURES:")
        print("- Automatic saving of cleaned files")
        print("- Optional loading of existing cleaned files") 
        print("- Force reprocessing option")
        print("- Configurable save format and directory")
        print("- File existence checking")
        
    except Exception as e:
        print(f"ERROR: {e}")
        
    print("=" * 80)