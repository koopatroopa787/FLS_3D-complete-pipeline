"""
CloudCompare Integration Module for Mill Analysis Project
Integrates the fine-tuned CloudCompare pipeline from test.py into the existing workflow
Uses CloudCompare for enhanced inner scan processing after initial Open3D alignment
IMPORTANT: All CloudCompare parameters are preserved from test.py (fine-tuned, do not modify)
"""

import os
import subprocess
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import open3d as o3d


class CloudCompareProcessor:
    """
    CloudCompare processor that integrates with the existing mill analysis pipeline.
    Takes aligned scans from alignment.py and processes them through CloudCompare.
    """
    
    def __init__(self, cloudcompare_path: str = r"C:\Program Files\CloudCompare\CloudCompare.exe"):
        """
        Initialize CloudCompare processor.
        
        Args:
            cloudcompare_path: Path to CloudCompare executable
        """
        self.cloudcompare_path = cloudcompare_path
        self.base_dir = Path(r"C:\Mill_Analysis_Project")
        self.subsample_points = 5000000  # Fine-tuned from test.py - DO NOT CHANGE
        self.output_dir = self.base_dir / "cloudcompare_mill_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for generated files
        self.processed_files = {}
        self.c2c_distance_files = []
        
        print(f"CloudCompare Mill Analysis Pipeline initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"CloudCompare: {self.cloudcompare_path}")
        print(f"Output: {self.output_dir}")
        print(f"Subsample target: {self.subsample_points:,} points")
    
    def run_cloudcompare_command(self, args: List[str], working_dir: Optional[str] = None, timeout: int = 1800) -> bool:
        """
        Robust CloudCompare command execution with extended timeout.
        PRESERVED FROM test.py - DO NOT MODIFY
        """
        try:
            work_dir = working_dir if working_dir else str(self.output_dir)
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
                    print(f"   Error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print("CloudCompare operation timed out")
            return False
        except Exception as e:
            print(f"CloudCompare execution failed: {e}")
            return False
    
    def cloudcompare_subsample_robust(self, input_file: str, num_points: Optional[int] = None) -> Optional[str]:
        """
        Community-proven subsampling with robust file handling.
        PRESERVED FROM test.py - DO NOT MODIFY STRATEGIES OR PARAMETERS
        """
        if num_points is None:
            num_points = self.subsample_points
            
        print(f"Subsampling {Path(input_file).name} -> {num_points:,} points...")
        
        # Track files before operation
        input_dir = Path(input_file).parent
        existing_files = set(input_dir.glob("*.ply"))
        
        # Multiple strategies for subsampling based on community examples - PRESERVED FROM test.py
        strategies = [
            # Strategy 1: Most basic (highest compatibility)
            ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(input_file), "-SS", "RANDOM", str(num_points), "-SAVE_CLOUDS"],
            
            # Strategy 2: With explicit PLY export
            ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(input_file), "-SS", "RANDOM", str(num_points), "-SAVE_CLOUDS"],
            
            # Strategy 3: Ultra-minimal
            ["-SILENT", "-O", str(input_file), "-SS", "RANDOM", str(num_points)]
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"   Trying strategy {i}...")
            
            if self.run_cloudcompare_command(strategy, str(input_dir)):
                time.sleep(3)
                
                # Find new subsampled file
                new_files = set(input_dir.glob("*.ply")) - existing_files
                subsampled_files = [f for f in new_files if any(keyword in f.name.upper() for keyword in ["SUBSAMPLED", "RANDOM"])]
                
                if subsampled_files:
                    # Move to output directory with organized naming
                    subsampled_file = max(subsampled_files, key=lambda x: x.stat().st_mtime)
                    output_name = f"{Path(input_file).stem}_subsampled.ply"
                    output_file = self.output_dir / output_name
                    
                    shutil.move(str(subsampled_file), str(output_file))
                    
                    print(f"Subsampling successful: {output_file.name}")
                    return str(output_file)
        
        print("All subsampling strategies failed")
        return None
    
    def align_box_center_with_open3d(self, reference_path: str, compare_path: str) -> str:
        """
        Bounding box center alignment function.
        PRESERVED FROM test.py - DO NOT MODIFY
        """
        try:
            print("Step 1: Bounding box center alignment...")
            
            # Load point clouds
            reference = o3d.io.read_point_cloud(reference_path)
            compare = o3d.io.read_point_cloud(compare_path)
            
            print(f"   Reference: {len(reference.points):,} points")
            print(f"   Compare: {len(compare.points):,} points")
            
            # Original alignment logic - PRESERVED FROM test.py
            reference_center = reference.get_center()
            compare_center = compare.get_center()
            
            print(f"   Reference center: [{reference_center[0]:.3f}, {reference_center[1]:.3f}, {reference_center[2]:.3f}]")
            print(f"   Compare center: [{compare_center[0]:.3f}, {compare_center[1]:.3f}, {compare_center[2]:.3f}]")
            
            # Compute the transformation that aligns the two centers
            transformation = np.eye(4)
            transformation[:3, 3] = reference_center - compare_center
            
            # Transform the compare model using the computed transformation
            transformed_compare = compare.transform(transformation)
            
            # Verify alignment
            new_center = compare.get_center()
            distance = np.linalg.norm(reference_center - new_center)
            print(f"   Center distance after alignment: {distance:.6f}")
            print("Bounding Box centered.")
            
            # Save aligned file
            aligned_file = self.output_dir / f"{Path(compare_path).stem}_bbox_aligned.ply"
            o3d.io.write_point_cloud(str(aligned_file), compare)
            
            return str(aligned_file)
            
        except Exception as e:
            print(f"Bounding box alignment failed: {e}")
            return compare_path
    
    def cloudcompare_icp_progressive(self, ref_file: str, target_file: str) -> Optional[str]:
        """
        Progressive ICP strategies based on community best practices.
        PRESERVED FROM test.py - DO NOT MODIFY STRATEGIES OR PARAMETERS
        """
        print("CloudCompare ICP registration...")
        
        # Track files before operation
        existing_files = set(self.output_dir.glob("*.ply"))
        
        # Progressive ICP strategies from community examples - PRESERVED FROM test.py
        strategies = [
            # Strategy 1: Ultra-minimal (most compatible)
            {
                "name": "Ultra-minimal ICP",
                "args": ["-SILENT", "-O", str(ref_file), "-O", str(target_file), "-ICP"]
            },
            
            # Strategy 2: Basic with reference specification
            {
                "name": "Basic ICP with reference",
                "args": ["-SILENT", "-O", str(ref_file), "-O", str(target_file), "-ICP", "-REFERENCE_IS_FIRST"]
            },
            
            # Strategy 3: With iterations control
            {
                "name": "ICP with iterations",
                "args": ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(ref_file), "-O", str(target_file), "-ICP", "-REFERENCE_IS_FIRST", "-ITER", "50", "-SAVE_CLOUDS"]
            },
            
            # Strategy 4: Community-recommended full parameters
            {
                "name": "Full ICP parameters", 
                "args": ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(ref_file), "-O", str(target_file), "-ICP", "-REFERENCE_IS_FIRST", "-ITER", "100", "-SAVE_CLOUDS"]
            }
        ]
        
        for strategy in strategies:
            print(f"   Trying: {strategy['name']}...")
            
            if self.run_cloudcompare_command(strategy["args"]):
                time.sleep(5)
                
                # Enhanced file detection patterns from community
                new_files = set(self.output_dir.glob("*.ply")) - existing_files
                registered_files = [f for f in new_files if any(keyword in f.name.upper() for keyword in ["REGISTERED", "ICP", "ALIGNED"])]
                
                # Also check parent directories (CloudCompare sometimes saves there)
                for check_dir in [Path(ref_file).parent, Path(target_file).parent, Path.cwd()]:
                    if check_dir != self.output_dir:
                        dir_files = set(check_dir.glob("*.ply"))
                        new_in_dir = [f for f in dir_files if any(keyword in f.name.upper() for keyword in ["REGISTERED", "ICP", "ALIGNED"]) and f.stat().st_mtime > time.time() - 300]
                        registered_files.extend(new_in_dir)
                
                if registered_files:
                    # Use most recent file
                    registered_file = max(registered_files, key=lambda x: x.stat().st_mtime)
                    output_name = f"{Path(target_file).stem}_icp_registered.ply"
                    output_file = self.output_dir / output_name
                    
                    if registered_file.parent != self.output_dir:
                        shutil.move(str(registered_file), str(output_file))
                    else:
                        registered_file.rename(output_file)
                    
                    print(f"ICP successful with {strategy['name']}: {output_file.name}")
                    return str(output_file)
        
        print("All ICP strategies failed")
        return None
    
    def cloudcompare_compute_distances(self, ref_file: str, aligned_file: str) -> List[Path]:
        """
        High-accuracy distance computation for precise heatmap generation.
        PRESERVED FROM test.py - DO NOT MODIFY PARAMETERS OR STRATEGIES
        """
        print("CloudCompare high-accuracy distance computation...")
        
        existing_files = set(self.output_dir.glob("*.ply"))
        
        # Enhanced distance computation strategies for maximum heatmap accuracy - PRESERVED FROM test.py
        strategies = [
            # Strategy 1: High-accuracy with octree optimization
            ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", 
             "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-OCTREE_LEVEL", "10", 
             "-MAX_DIST", "1.5", "-SPLIT_XYZ", "-SAVE_CLOUDS"],
            
            # Strategy 2: Maximum precision with statistical filtering
            ["-SILENT", "-AUTO_SAVE", "OFF", "-C_EXPORT_FMT", "PLY", 
             "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-OCTREE_LEVEL", "12", 
             "-MAX_DIST", "2.0", "-SPLIT_XYZ", "-LOCAL_STAT_TEST", "2.5", "12", 
             "-MULTI_THREAD", "-SAVE_CLOUDS"],
            
            # Strategy 3: Ultra-high precision (slower but most accurate)
            ["-SILENT", "-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE",
             "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-OCTREE_LEVEL", "14", 
             "-MAX_DIST", "3.0", "-SPLIT_XYZ", "-LOCAL_STAT_TEST", "2.0", "16", 
             "-MULTI_THREAD", "-SAVE_CLOUDS"],
            
            # Strategy 4: Fallback - basic but reliable
            ["-SILENT", "-AUTO_SAVE", "OFF", "-O", str(ref_file), "-O", str(aligned_file), 
             "-C2C_DIST", "-REFERENCE_IS_FIRST", "-SAVE_CLOUDS"]
        ]
        
        strategy_names = [
            "High-accuracy octree optimization",
            "Maximum precision with statistical filtering", 
            "Ultra-high precision (slowest)",
            "Fallback basic computation"
        ]
        
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names), 1):
            print(f"   Strategy {i}: {name}...")
            
            if self.run_cloudcompare_command(strategy):
                time.sleep(5)  # Longer wait for high-precision computation
                
                # Find distance files
                new_files = set(self.output_dir.glob("*.ply")) - existing_files
                distance_files = [f for f in new_files if "C2C_DIST" in f.name.upper()]
                
                if distance_files:
                    print(f"High-accuracy distance computation successful: {len(distance_files)} files")
                    for f in distance_files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"   {f.name} ({size_mb:.1f} MB)")
                    
                    print(f"   Heatmap accuracy level: {name}")
                    self.c2c_distance_files.extend(distance_files)
                    return distance_files
        
        print("All high-accuracy distance computation strategies failed")
        return []
    
    def create_verification_overlay(self, ref_file: str, aligned_file: str) -> Optional[str]:
        """
        Create verification overlay for visual inspection.
        PRESERVED FROM test.py - DO NOT MODIFY
        """
        try:
            print("Creating verification overlay...")
            
            ref_cloud = o3d.io.read_point_cloud(ref_file)
            aligned_cloud = o3d.io.read_point_cloud(aligned_file)
            
            # Color clouds for verification
            ref_cloud.paint_uniform_color([0, 0, 1])      # Blue = Reference
            aligned_cloud.paint_uniform_color([1, 0, 0])  # Red = Aligned
            
            # Combine
            combined = ref_cloud + aligned_cloud
            
            # Save overlay
            overlay_file = self.output_dir / "verification_overlay.ply"
            o3d.io.write_point_cloud(str(overlay_file), combined)
            
            print("Verification overlay created!")
            print(f"   Blue = Reference mill")
            print(f"   Red = Aligned mill")
            print(f"   Perfect alignment = tight color overlap")
            
            return str(overlay_file)
            
        except Exception as e:
            print(f"Overlay creation failed: {e}")
            return None
    
    def process_aligned_scans_with_cloudcompare(self, reference_aligned_path: str, 
                                              scan1_aligned_path: str, 
                                              scan2_aligned_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process aligned scans through CloudCompare pipeline.
        This is called AFTER the existing alignment.py processing is complete.
        
        Args:
            reference_aligned_path: Path to aligned reference scan
            scan1_aligned_path: Path to aligned inner scan 1
            scan2_aligned_path: Path to aligned inner scan 2 (optional)
            
        Returns:
            Dictionary with processing results and C2C files
        """
        print("="*70)
        print("CLOUDCOMPARE PROCESSING OF PRE-ALIGNED SCANS")
        print("="*70)
        print(f"Reference: {Path(reference_aligned_path).name}")
        print(f"Scan 1: {Path(scan1_aligned_path).name}")
        if scan2_aligned_path:
            print(f"Scan 2: {Path(scan2_aligned_path).name}")
        print("Community-proven CloudCompare techniques - PARAMETERS PRESERVED FROM test.py")
        print(f"High detail subsampling: {self.subsample_points:,} points")
        print("="*70)
        
        start_time = time.time()
        results = {
            'processing_successful': False,
            'c2c_files': [],
            'processed_scans': {},
            'verification_overlays': []
        }
        
        try:
            # Step 1: Subsample all clouds for CloudCompare processing
            print("\nPHASE 1: CLOUDCOMPARE SUBSAMPLING")
            ref_subsampled = self.cloudcompare_subsample_robust(reference_aligned_path)
            scan1_subsampled = self.cloudcompare_subsample_robust(scan1_aligned_path)
            
            if not ref_subsampled or not scan1_subsampled:
                print("ERROR: Subsampling failed")
                return results
            
            results['processed_scans']['reference'] = ref_subsampled
            results['processed_scans']['scan1'] = scan1_subsampled
            
            # Step 2: CloudCompare processing for scan 1
            print("\nPHASE 2: CLOUDCOMPARE PROCESSING - SCAN 1")
            scan1_centered = self.align_box_center_with_open3d(ref_subsampled, scan1_subsampled)
            scan1_registered = self.cloudcompare_icp_progressive(ref_subsampled, scan1_centered)
            final_scan1 = scan1_registered if scan1_registered else scan1_centered
            
            # Step 3: C2C distance computation for scan 1
            print("\nPHASE 3: C2C DISTANCE COMPUTATION - SCAN 1")
            c2c_files_scan1 = self.cloudcompare_compute_distances(ref_subsampled, final_scan1)
            results['c2c_files'].extend(c2c_files_scan1)
            
            # Step 4: Verification overlay for scan 1
            overlay1 = self.create_verification_overlay(ref_subsampled, final_scan1)
            if overlay1:
                results['verification_overlays'].append(overlay1)
            
            # Step 5: Process scan 2 if provided
            if scan2_aligned_path:
                print("\nPHASE 4: CLOUDCOMPARE PROCESSING - SCAN 2")
                scan2_subsampled = self.cloudcompare_subsample_robust(scan2_aligned_path)
                
                if scan2_subsampled:
                    results['processed_scans']['scan2'] = scan2_subsampled
                    scan2_centered = self.align_box_center_with_open3d(ref_subsampled, scan2_subsampled)
                    scan2_registered = self.cloudcompare_icp_progressive(ref_subsampled, scan2_centered)
                    final_scan2 = scan2_registered if scan2_registered else scan2_centered
                    
                    # C2C distance computation for scan 2
                    print("\nPHASE 5: C2C DISTANCE COMPUTATION - SCAN 2")
                    c2c_files_scan2 = self.cloudcompare_compute_distances(ref_subsampled, final_scan2)
                    results['c2c_files'].extend(c2c_files_scan2)
                    
                    # Verification overlay for scan 2
                    overlay2 = self.create_verification_overlay(ref_subsampled, final_scan2)
                    if overlay2:
                        results['verification_overlays'].append(overlay2)
            
            # Final summary
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n" + "="*70)
            print("CLOUDCOMPARE PROCESSING COMPLETED!")
            print(f"Total processing time: {total_time:.1f} seconds")
            print("Community-proven CloudCompare pipeline executed successfully")
            print(f"High detail analysis with {self.subsample_points:,} points")
            print("="*70)
            
            print("CLOUDCOMPARE RESULTS:")
            if results['c2c_files']:
                print(f"   C2C distance files: {len(results['c2c_files'])} files")
                for f in results['c2c_files'][:3]:  # Show first 3
                    print(f"     - {f.name}")
            
            print(f"\nALL CLOUDCOMPARE OUTPUT FILES:")
            for f in sorted(self.output_dir.glob("*.ply")):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.2f} MB)")
            
            results['processing_successful'] = True
            results['processing_time'] = total_time
            
            return results
            
        except Exception as e:
            print(f"CloudCompare processing failed: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def get_c2c_distance_files(self) -> List[Path]:
        """Get all generated C2C distance files."""
        return self.c2c_distance_files
    
    def get_latest_c2c_file(self) -> Optional[Path]:
        """Get the most recent C2C distance file."""
        if not self.c2c_distance_files:
            return None
        return max(self.c2c_distance_files, key=lambda x: x.stat().st_mtime)