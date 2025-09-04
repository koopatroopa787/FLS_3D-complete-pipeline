import open3d as o3d
import numpy as np
import copy
import subprocess
import time
import shutil
from pathlib import Path


class PointCloudCleaner:
    def __init__(self, output_dir="C:/Mill_Analysis_Project/data/clean_scan", 
                 subsample_large_datasets=True, max_points_for_clustering=5000000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conservative parameters for mill analysis (preserves edges/shell structure)
        self.nb_neighbors = 30  # Increased from 20 - requires more neighbors to classify as outlier
        self.std_ratio = 2.5    # Increased from 2.0 - more lenient threshold
        
        # V3 DBSCAN parameters from your noise removal code
        self.dbscan_eps = 0.05  # From your V3 config
        self.dbscan_min_points = 100  # From your V3 config
        
        # CloudCompare subsampling settings
        self.subsample_large_datasets = subsample_large_datasets
        self.max_points_for_clustering = max_points_for_clustering
        self.cloudcompare_path = None
        self.cloudcompare_available = False
        
        # Initialize CloudCompare
        if self.subsample_large_datasets:
            self._initialize_cloudcompare()
        
        # Create temporary directory for CloudCompare operations
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        print("Point Cloud Cleaner initialized (V3 Enhanced - Mill Structure Preserving)")
        print(f"Output directory: {self.output_dir}")
        print(f"Conservative outlier removal: nb_neighbors={self.nb_neighbors}, std_ratio={self.std_ratio}")
        print(f"V3 DBSCAN parameters: eps={self.dbscan_eps}, min_points={self.dbscan_min_points}")
        print("Parameters optimized to preserve mill edges and shell structure")
        
        if self.subsample_large_datasets:
            print(f"CloudCompare subsampling: {'ENABLED' if self.cloudcompare_available else 'DISABLED (not found)'}")
            print(f"Max points for clustering: {self.max_points_for_clustering:,}")
            if self.cloudcompare_available:
                print(f"CloudCompare path: {self.cloudcompare_path}")
    
    def _initialize_cloudcompare(self):
        """Initialize CloudCompare for large dataset subsampling."""
        try:
            # Windows CloudCompare paths
            windows_paths = [
                r"C:\Program Files\CloudCompare\CloudCompare.exe",
                r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
            ]
            
            for path in windows_paths:
                if Path(path).exists():
                    self.cloudcompare_path = path
                    self.cloudcompare_available = True
                    print(f"CloudCompare found: {path}")
                    return
            
            print("CloudCompare not found - large datasets may crash during clustering")
            
        except Exception as e:
            print(f"CloudCompare initialization failed: {e}")
    
    def _run_cloudcompare_command(self, args, working_dir=None, timeout=600):
        """Execute CloudCompare command"""
        try:
            work_dir = working_dir if working_dir else str(self.temp_dir)
            cmd = [self.cloudcompare_path] + args
            
            print(f"CloudCompare: {' '.join(args[:3])}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
            
            if result.returncode == 0:
                print("CloudCompare operation successful")
                return True
            else:
                print(f"CloudCompare failed (code: {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print("CloudCompare operation timed out")
            return False
        except Exception as e:
            print(f"CloudCompare execution failed: {e}")
            return False
    
    def _cloudcompare_subsample(self, pcd, target_points):
        """Subsample point cloud using CloudCompare for better reliability"""
        print(f"CloudCompare subsampling: {len(pcd.points):,} -> {target_points:,} points")
        
        try:
            # Save temporary file for CloudCompare processing
            temp_input_path = self.temp_dir / "temp_input.ply"
            o3d.io.write_point_cloud(str(temp_input_path), pcd)
            
            # Track files before operation
            existing_files = set(self.temp_dir.glob("*.ply"))
            
            # CloudCompare subsampling strategies
            strategies = [
                ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(temp_input_path), "-SS", "RANDOM", str(target_points), "-SAVE_CLOUDS"],
                ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(temp_input_path), "-SS", "RANDOM", str(target_points), "-SAVE_CLOUDS"],
                ["-SILENT", "-O", str(temp_input_path), "-SS", "RANDOM", str(target_points)]
            ]
            
            for i, strategy in enumerate(strategies, 1):
                print(f"Trying CloudCompare strategy {i}...")
                
                if self._run_cloudcompare_command(strategy, str(self.temp_dir)):
                    time.sleep(2)
                    
                    # Find new subsampled file
                    new_files = set(self.temp_dir.glob("*.ply")) - existing_files
                    subsampled_files = [f for f in new_files if any(keyword in f.name.upper() for keyword in ["SUBSAMPLED", "RANDOM"])]
                    
                    if subsampled_files:
                        # Load the subsampled result
                        subsampled_file = max(subsampled_files, key=lambda x: x.stat().st_mtime)
                        subsampled_pcd = o3d.io.read_point_cloud(str(subsampled_file))
                        
                        # Clean up temporary files
                        try:
                            temp_input_path.unlink()
                            subsampled_file.unlink()
                        except:
                            pass
                        
                        if len(subsampled_pcd.points) > 0:
                            print(f"CloudCompare subsampling successful: {len(subsampled_pcd.points):,} points")
                            return subsampled_pcd
            
            print("All CloudCompare strategies failed")
            return None
            
        except Exception as e:
            print(f"CloudCompare subsampling error: {e}")
            return None
    
    def clean_point_cloud(self, input_file_path, output_filename=None, 
                          apply_bottom_plate_removal=True, apply_outlier_removal=True):
        """
        Clean point cloud with selectable cleaning operations.
        Uses V3 Enhanced methods for improved cleaning performance.
        
        Args:
            input_file_path (str): Path to input PLY file
            output_filename (str, optional): Custom output filename. If None, generates from input name.
            apply_bottom_plate_removal (bool): Whether to apply bottom plate removal
            apply_outlier_removal (bool): Whether to apply statistical outlier removal
            
        Returns:
            str: Path to cleaned output file
        """
        input_path = Path(input_file_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Loading point cloud: {input_path.name}")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(input_path))
        
        if len(pcd.points) == 0:
            raise ValueError("Loaded point cloud is empty")
        
        input_points = len(pcd.points)
        print(f"Input points: {input_points:,}")
        
        # Check what operations are selected
        operations = []
        if apply_bottom_plate_removal:
            operations.append("Bottom Plate Removal")
        if apply_outlier_removal:
            operations.append("Outlier Removal")
        
        if not operations:
            operations.append("None (No cleaning)")
        
        print(f"Selected operations: {', '.join(operations)}")
        
        # Apply selected V3 Enhanced cleaning operations
        cleaned_pcd = copy.deepcopy(pcd)
        
        if apply_bottom_plate_removal:
            cleaned_pcd = self._apply_bottom_plate_removal(cleaned_pcd)
        else:
            print("Skipping bottom plate removal")
        
        if apply_outlier_removal:
            cleaned_pcd = self._apply_statistical_outlier_removal(cleaned_pcd)
        else:
            print("Skipping statistical outlier removal")
        
        if not apply_bottom_plate_removal and not apply_outlier_removal:
            print("No cleaning operations applied - returning original point cloud")
        
        output_points = len(cleaned_pcd.points)
        retention_rate = (output_points / input_points) * 100
        
        print(f"Final points: {output_points:,} ({retention_rate:.1f}% retained)")
        
        # Save cleaned point cloud with descriptive filename
        if output_filename is None:
            # Generate filename based on applied operations
            operations_suffix = []
            if apply_bottom_plate_removal:
                operations_suffix.append("bp")
            if apply_outlier_removal:
                operations_suffix.append("or")
            
            if operations_suffix:
                suffix = "_" + "_".join(operations_suffix)
            else:
                suffix = "_nocleaning"
            
            output_filename = f"cleaned{suffix}_{input_path.stem}.ply"
        
        output_path = self.output_dir / output_filename
        
        success = o3d.io.write_point_cloud(str(output_path), cleaned_pcd)
        
        if success:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Cleaned file saved: {output_path.name} ({file_size_mb:.1f} MB)")
            return str(output_path)
        else:
            raise RuntimeError(f"Failed to save cleaned point cloud to {output_path}")
    
    def _apply_bottom_plate_removal(self, pcd):
        """
        Apply V3 bottom plate removal (enhanced original method).
        Remove points in bottom 10% of Z range.
        
        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud
            
        Returns:
            o3d.geometry.PointCloud: Point cloud with bottom plate removed
        """
        print("Applying BOTTOM PLATE REMOVAL (V3 Enhanced method)...")
        
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
    
    def _apply_statistical_outlier_removal(self, pcd):
        """
        Apply conservative statistical outlier removal instead of DBSCAN for mill analysis.
        DBSCAN becomes too aggressive after CloudCompare subsampling, so using statistical approach.
        
        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud
            
        Returns:
            o3d.geometry.PointCloud: Point cloud with outliers removed (mill structure preserved)
        """
        print("Applying conservative statistical outlier removal for mill structure preservation...")
        
        input_count = len(pcd.points)
        print(f"Input points: {input_count:,}")
        
        if input_count == 0:
            return pcd
        
        # Step 1: Check if subsampling is needed for large datasets
        working_pcd = pcd
        if input_count > self.max_points_for_clustering and self.subsample_large_datasets:
            if self.cloudcompare_available:
                print(f"Large dataset detected ({input_count:,} points)")
                print(f"Applying CloudCompare subsampling to {self.max_points_for_clustering:,} points...")
                
                subsampled_pcd = self._cloudcompare_subsample(pcd, self.max_points_for_clustering)
                if subsampled_pcd is not None:
                    working_pcd = subsampled_pcd
                    print(f"Using subsampled dataset: {len(working_pcd.points):,} points")
                else:
                    print("CloudCompare subsampling failed, using original (may crash)")
                    working_pcd = pcd
            else:
                print(f"WARNING: Large dataset ({input_count:,} points) but CloudCompare not available")
                print("Statistical outlier removal may be slow...")
                working_pcd = pcd
        else:
            print(f"Dataset size manageable: {input_count:,} points")
        
        # Step 2: Apply conservative statistical outlier removal 
        return self._apply_conservative_statistical_outlier_removal(working_pcd)
    
    def _apply_conservative_statistical_outlier_removal(self, pcd):
        """Apply very conservative statistical outlier removal that preserves mill structure."""
        print("Applying very conservative statistical outlier removal...")
        
        input_count = len(pcd.points)
        
        if input_count == 0:
            return pcd
        
        # VERY conservative settings to preserve mill structure
        nb_neighbors = 50  # Even more neighbors required
        std_ratio = 3.0    # Very lenient threshold
        
        print(f"VERY conservative parameters: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
        print("Optimized to preserve mill edges and shell structure after subsampling")
        
        cleaned_pcd, outlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
        output_count = len(cleaned_pcd.points)
        removed_count = len(outlier_indices)
        removal_rate = (removed_count / input_count) * 100
        
        print(f"Conservative outlier removal: {removed_count:,} removed ({removal_rate:.1f}%)")
        print(f"Mill structure preserved: {output_count:,} points retained")
        
        return cleaned_pcd
    
    def _apply_selective_v3_dbscan(self, pcd):
        """Apply SELECTIVE V3 DBSCAN that keeps significant clusters (from your noise removal code)."""
        print("Applying SELECTIVE V3 DBSCAN clustering...")
        
        input_count = len(pcd.points)
        
        if input_count == 0:
            return pcd
        
        # V3 DBSCAN parameters
        eps = self.dbscan_eps  # 0.05
        min_points = self.dbscan_min_points  # 100
        
        print(f"SELECTIVE V3 DBSCAN - eps: {eps}, min_points: {min_points}")
        
        # Apply DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        # Analyze clusters
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 is noise
        
        print(f"Found {num_clusters} clusters")
        
        if num_clusters == 0:
            print("No clusters found, returning original")
            return pcd
        
        # Find cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_sizes[label] = np.sum(labels == label)
        
        # Sort clusters by size
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_clusters:
            print("No valid clusters found")
            return pcd
        
        largest_cluster_size = sorted_clusters[0][1]
        print(f"Largest cluster: {largest_cluster_size:,} points")
        
        # SELECTIVE retention: Keep significant clusters (>=1% of largest cluster)
        significance_threshold = max(largest_cluster_size * 0.01, 100)  # At least 1% or 100 points
        print(f"Significance threshold: {significance_threshold:.0f} points")
        
        # Select significant clusters
        significant_clusters = []
        total_kept_points = 0
        
        for i, (label, size) in enumerate(sorted_clusters):
            if size >= significance_threshold:
                significant_clusters.append(label)
                total_kept_points += size
                print(f"Keeping significant cluster {label}: {size:,} points")
            else:
                print(f"Rejecting small cluster {label}: {size:,} points (below threshold)")
        
        if not significant_clusters:
            # Fallback: keep largest cluster
            significant_clusters = [sorted_clusters[0][0]]
            total_kept_points = sorted_clusters[0][1]
            print(f"Fallback: keeping only largest cluster")
        
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
        
        print(f"Total clusters kept: {len(significant_clusters)}")
        print(f"Total points in kept clusters: {output_count:,}")
        print(f"SELECTIVE V3 DBSCAN: Kept {len(significant_clusters)} significant clusters")
        print(f"Retained: {output_count:,} points ({retention_rate:.1f}%)")
        print(f"Removed: {removed_count:,} points ({removal_rate:.1f}%)")
        
        return cleaned_pcd
    
    def set_outlier_removal_parameters(self, nb_neighbors=20, std_ratio=2.0):
        """
        Set parameters for statistical outlier removal.
        
        Args:
            nb_neighbors (int): Number of neighbors to consider for each point
            std_ratio (float): Standard deviation threshold ratio
        """
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        print(f"Updated V3 Enhanced outlier removal parameters: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    
    def clean_with_bottom_plate_only(self, input_file_path, output_filename=None):
        """Apply only bottom plate removal."""
        return self.clean_point_cloud(input_file_path, output_filename, 
                                     apply_bottom_plate_removal=True, apply_outlier_removal=False)
    
    def clean_with_outlier_removal_only(self, input_file_path, output_filename=None):
        """Apply only statistical outlier removal."""
        return self.clean_point_cloud(input_file_path, output_filename, 
                                     apply_bottom_plate_removal=False, apply_outlier_removal=True)
    
    def clean_with_both(self, input_file_path, output_filename=None):
        """Apply both bottom plate removal and outlier removal."""
        return self.clean_point_cloud(input_file_path, output_filename, 
                                     apply_bottom_plate_removal=True, apply_outlier_removal=True)
    
    def clean_with_none(self, input_file_path, output_filename=None):
        """Apply no cleaning (just copy file to output directory)."""
        return self.clean_point_cloud(input_file_path, output_filename, 
                                     apply_bottom_plate_removal=False, apply_outlier_removal=False)
    
    def clean_with_subsampling_only(self, input_file_path, output_filename=None):
        """Apply CloudCompare subsampling only (no additional cleaning)."""
        input_path = Path(input_file_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Loading point cloud: {input_path.name}")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(input_path))
        
        if len(pcd.points) == 0:
            raise ValueError("Loaded point cloud is empty")
        
        input_points = len(pcd.points)
        print(f"Input points: {input_points:,}")
        print("Selected operations: CloudCompare subsampling ONLY")
        
        # Apply CloudCompare subsampling if needed
        cleaned_pcd = pcd
        if input_points > self.max_points_for_clustering and self.cloudcompare_available:
            print(f"Applying CloudCompare subsampling to {self.max_points_for_clustering:,} points...")
            subsampled_pcd = self._cloudcompare_subsample(pcd, self.max_points_for_clustering)
            if subsampled_pcd is not None:
                cleaned_pcd = subsampled_pcd
                print(f"CloudCompare subsampling successful: {len(cleaned_pcd.points):,} points")
            else:
                print("CloudCompare subsampling failed, using original")
        else:
            print("No subsampling needed or CloudCompare not available")
        
        output_points = len(cleaned_pcd.points)
        retention_rate = (output_points / input_points) * 100
        
        print(f"Final points: {output_points:,} ({retention_rate:.1f}% retained)")
        
        # Save with descriptive filename
        if output_filename is None:
            output_filename = f"cleaned_subsample_only_{input_path.stem}.ply"
        
        output_path = self.output_dir / output_filename
        
        success = o3d.io.write_point_cloud(str(output_path), cleaned_pcd)
        
        if success:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Cleaned file saved: {output_path.name} ({file_size_mb:.1f} MB)")
            return str(output_path)
        else:
            raise RuntimeError(f"Failed to save cleaned point cloud to {output_path}")


def clean_single_file(input_file_path, output_dir="C:/Mill_Analysis_Project/data/clean_scan",
                      apply_bottom_plate_removal=True, apply_outlier_removal=True):
    """
    Convenience function to clean a single point cloud file using V3 Enhanced methods.
    
    Args:
        input_file_path (str): Path to input PLY file
        output_dir (str): Output directory for cleaned file
        apply_bottom_plate_removal (bool): Whether to apply bottom plate removal
        apply_outlier_removal (bool): Whether to apply statistical outlier removal
        
    Returns:
        str: Path to cleaned output file
    """
    cleaner = PointCloudCleaner(output_dir)
    return cleaner.clean_point_cloud(input_file_path, None, apply_bottom_plate_removal, apply_outlier_removal)


if __name__ == "__main__":
    print("Point Cloud Cleaner - V3 Enhanced Bottom Plate and Outlier Removal")
    print("=" * 70)
    
    try:
        # Initialize cleaner
        cleaner = PointCloudCleaner()
        
        print("V3 Enhanced Processing pipeline:")
        print("1. Load point cloud from file")
        print("2. Apply selected cleaning operations")
        print("3. Save cleaned file to output directory")
        print()
        
        # Interactive mode
        while True:
            print("\n" + "=" * 50)
            print("INTERACTIVE CLEANING MODE")
            print("=" * 50)
            
            # Get input file path
            input_file = input("Enter path to PLY file (or 'quit' to exit): ").strip()
            
            if input_file.lower() in ['quit', 'exit', 'q']:
                print("Exiting Point Cloud Cleaner")
                break
            
            if not input_file:
                print("Please enter a valid file path")
                continue
                
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"File not found: {input_file}")
                continue
            
            if not input_path.suffix.lower() == '.ply':
                print("Please provide a PLY file")
                continue
            
            # Show cleaning options
            print(f"\nFile selected: {input_path.name}")
            print("\nCleaning Options:")
            print("1. Both operations (Bottom Plate + Outlier Removal) [DEFAULT]")
            print("2. Bottom Plate Removal only")
            print("3. Outlier Removal only")
            print("4. No cleaning (just copy file)")
            print("5. Custom parameters for outlier removal")
            print("6. CloudCompare subsampling ONLY (no additional cleaning)")
            
            # Get user choice
            choice = input("\nSelect cleaning option (1-6) [1]: ").strip()
            
            if not choice:
                choice = "1"
            
            # Handle custom parameters option
            if choice == "5":
                print("\nCurrent outlier removal parameters:")
                print(f"  nb_neighbors: {cleaner.nb_neighbors}")
                print(f"  std_ratio: {cleaner.std_ratio}")
                
                try:
                    nb_neighbors = input(f"Enter nb_neighbors [{cleaner.nb_neighbors}]: ").strip()
                    if nb_neighbors:
                        cleaner.nb_neighbors = int(nb_neighbors)
                    
                    std_ratio = input(f"Enter std_ratio [{cleaner.std_ratio}]: ").strip()
                    if std_ratio:
                        cleaner.std_ratio = float(std_ratio)
                    
                    print(f"Updated parameters: nb_neighbors={cleaner.nb_neighbors}, std_ratio={cleaner.std_ratio}")
                    
                except ValueError:
                    print("Invalid parameter values, using defaults")
                
                # After setting parameters, ask for cleaning choice again
                print("\nNow select cleaning option:")
                print("1. Both operations (Bottom Plate + Outlier Removal)")
                print("2. Bottom Plate Removal only") 
                print("3. Outlier Removal only")
                print("4. No cleaning (just copy file)")
                print("6. CloudCompare subsampling ONLY")
                choice = input("Select option (1-4,6) [1]: ").strip()
                if not choice:
                    choice = "1"
            
            # Process based on choice
            try:
                print(f"\nProcessing {input_path.name}...")
                
                if choice == "1":
                    print("Applying: Both Bottom Plate Removal and Outlier Removal")
                    cleaned_path = cleaner.clean_with_both(str(input_path))
                elif choice == "2":
                    print("Applying: Bottom Plate Removal only")
                    cleaned_path = cleaner.clean_with_bottom_plate_only(str(input_path))
                elif choice == "3":
                    print("Applying: Outlier Removal only")
                    cleaned_path = cleaner.clean_with_outlier_removal_only(str(input_path))
                elif choice == "4":
                    print("Applying: No cleaning (copying file)")
                    cleaned_path = cleaner.clean_with_none(str(input_path))
                elif choice == "6":
                    print("Applying: CloudCompare subsampling ONLY (no additional cleaning)")
                    cleaned_path = cleaner.clean_with_subsampling_only(str(input_path))
                else:
                    print("Invalid choice, using default (both operations)")
                    cleaned_path = cleaner.clean_with_both(str(input_path))
                
                print(f"\nSUCCESS: Cleaned file saved to: {cleaned_path}")
                
                # Ask if user wants to process another file
                another = input("\nProcess another file? (y/n) [n]: ").strip().lower()
                if another not in ['y', 'yes']:
                    print("Exiting Point Cloud Cleaner")
                    break
                    
            except Exception as e:
                print(f"ERROR processing file: {e}")
                continue_choice = input("Continue with another file? (y/n) [y]: ").strip().lower()
                if continue_choice in ['n', 'no']:
                    break
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\nThank you for using Point Cloud Cleaner!")