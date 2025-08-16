"""
Professional Data Loader for Mill Analysis Project - PURE DATA LOADING ONLY
Open3D implementation - Split from original massive file
Handles file loading, validation, and basic analysis only
"""

import open3d as o3d
import numpy as np
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from config import get_file_paths, ProcessingConfig, FileConfig


class PointCloudLoader:
    """Professional point cloud loader using Open3D with memory management."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.file_paths = get_file_paths()
        
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate that a file exists and is readable.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return False
            
        if not file_path.is_file():
            print(f"ERROR: Path is not a file: {file_path}")
            return False
            
        # Check file extension
        if file_path.suffix.lower() not in FileConfig.SUPPORTED_FORMATS:
            print(f"ERROR: Unsupported file format: {file_path.suffix}")
            print(f"Supported formats: {FileConfig.SUPPORTED_FORMATS}")
            return False
            
        # Check file size (basic check)
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb == 0:
                print(f"ERROR: File is empty: {file_path}")
                return False
                
            print(f"File validation passed: {file_path.name} ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"ERROR: Cannot access file {file_path}: {e}")
            return False
    
    def get_point_cloud_info(self, pcd: o3d.geometry.PointCloud, name: str = "Point Cloud") -> Dict[str, Any]:
        """
        Get comprehensive information about a point cloud.
        
        Args:
            pcd: Open3D point cloud
            name: Descriptive name for the point cloud
            
        Returns:
            Dictionary with point cloud analysis
        """
        if pcd is None or len(pcd.points) == 0:
            return {'name': name, 'valid': False, 'error': 'Empty point cloud'}
            
        points = np.asarray(pcd.points)
        
        # Basic statistics
        center = pcd.get_center()
        bbox = pcd.get_axis_aligned_bounding_box()
        dimensions = bbox.get_extent()
        
        # Calculate characteristic radius (95th percentile method from all versions)
        radial_distances = np.sqrt(np.sum((points - center)**2, axis=1))
        char_radius = np.percentile(radial_distances, 95)
        
        # Point cloud properties
        has_colors = pcd.has_colors()
        has_normals = pcd.has_normals()
        
        analysis = {
            'name': name,
            'valid': True,
            'num_points': len(points),
            'center': center.tolist(),
            'dimensions': dimensions.tolist(),
            'max_dimension': float(np.max(dimensions)),
            'characteristic_radius': float(char_radius),
            'has_colors': has_colors,
            'has_normals': has_normals,
            'bbox_min': bbox.min_bound.tolist(),
            'bbox_max': bbox.max_bound.tolist()
        }
        
        return analysis
    
    def print_point_cloud_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted point cloud analysis."""
        if not analysis.get('valid', False):
            print(f"INVALID: {analysis.get('name', 'Unknown')} - {analysis.get('error', 'Unknown error')}")
            return
            
        print(f"\n=== {analysis['name']} Analysis ===")
        print(f"Points: {analysis['num_points']:,}")
        print(f"Center: [{analysis['center'][0]:.2f}, {analysis['center'][1]:.2f}, {analysis['center'][2]:.2f}]")
        print(f"Dimensions: [{analysis['dimensions'][0]:.2f}, {analysis['dimensions'][1]:.2f}, {analysis['dimensions'][2]:.2f}]")
        print(f"Max dimension: {analysis['max_dimension']:.2f}")
        print(f"Characteristic radius: {analysis['characteristic_radius']:.2f}")
        print(f"Has colors: {analysis['has_colors']}")
        print(f"Has normals: {analysis['has_normals']}")
    
    def load_point_cloud_file(self, file_path: Path, 
                             preserve_properties: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """
        Load point cloud file directly without any downsampling (V3 flow).
        
        Args:
            file_path: Path to point cloud file
            preserve_properties: Whether to compute/preserve normals and colors
            
        Returns:
            Loaded point cloud or None if failed
        """
        if not self.validate_file(file_path):
            return None
            
        print(f"\nLoading point cloud: {file_path.name}")
        start_time = time.time()
        
        try:
            # Load point cloud using Open3D (V3 exact flow)
            pcd = o3d.io.read_point_cloud(str(file_path))
            
            if len(pcd.points) == 0:
                print(f"ERROR: No points found in {file_path.name}")
                return None
                
            points_count = len(pcd.points)
            print(f"Loaded {points_count:,} points (no downsampling - V3 flow)")
            
            # Preserve/compute properties if requested
            if preserve_properties:
                # Compute normals if not present
                if not pcd.has_normals():
                    print("Computing normals...")
                    pcd.estimate_normals()
                    
                # Colors are preserved automatically if present
                if pcd.has_colors():
                    print("Colors preserved from file")
            
            load_time = time.time() - start_time
            print(f"Loading completed in {load_time:.2f} seconds")
            
            return pcd
            
        except Exception as e:
            print(f"ERROR loading {file_path.name}: {str(e)}")
            return None
    
    def load_reference_file(self) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """
        Load reference file directly (V3 exact flow - no downsampling).
        
        Returns:
            Tuple of (point_cloud, analysis_dict)
        """
        print("\n" + "="*80)
        print("LOADING REFERENCE FILE (V3 FLOW - NO DOWNSAMPLING)")
        print("="*80)
        
        ref_path = self.file_paths['reference']
        
        # Load directly without any downsampling (V3 flow)
        pcd = self.load_point_cloud_file(ref_path, preserve_properties=True)
        
        if pcd is None:
            return None, {}
            
        # Analyze reference point cloud
        analysis = self.get_point_cloud_info(pcd, "Reference Shell")
        self.print_point_cloud_analysis(analysis)
        
        print(f"\nREFERENCE LOADED SUCCESSFULLY (V3 FLOW):")
        print(f"  Points loaded: {analysis['num_points']:,}")
        print(f"  No downsampling applied")
        print(f"  Ready for processing pipeline")
        
        return pcd, analysis
    
    def load_inner_scan_file(self) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """
        Load inner scan file directly (V3 exact flow - no downsampling in loading phase).
        Downsampling will happen later in the noise removal process.
        
        Returns:
            Tuple of (point_cloud, analysis_dict)
        """
        print("\n" + "="*80)
        print("LOADING INNER SCAN FILE (V3 FLOW - FULL RESOLUTION)")
        print("="*80)
        
        scan_path = self.file_paths['inner_scan']
        
        # Load directly without any downsampling (V3 flow)
        # Downsampling will happen in the noise removal voxel step
        pcd = self.load_point_cloud_file(scan_path, preserve_properties=True)
        
        if pcd is None:
            return None, {}
            
        # Analyze inner scan point cloud
        analysis = self.get_point_cloud_info(pcd, "Inner Scan")
        self.print_point_cloud_analysis(analysis)
        
        print(f"\nINNER SCAN LOADED SUCCESSFULLY (V3 FLOW):")
        print(f"  Points loaded: {analysis['num_points']:,}")
        print(f"  Full resolution preserved")
        print(f"  Downsampling will occur in noise removal voxel step")
        print(f"  Ready for V3 noise removal pipeline")
        
        return pcd, analysis
    
    def load_both_files(self) -> Tuple[Optional[o3d.geometry.PointCloud], 
                                     Optional[o3d.geometry.PointCloud],
                                     Dict[str, Any]]:
        """
        Load both reference and inner scan files with appropriate handling.
        
        Returns:
            Tuple of (reference_pcd, inner_scan_pcd, combined_analysis)
        """
        print("="*100)
        print("MILL ANALYSIS - LOADING BOTH DATA FILES")
        print("="*100)
        
        # Load reference file (no downsampling)
        ref_pcd, ref_analysis = self.load_reference_file()
        
        if ref_pcd is None:
            print("CRITICAL ERROR: Failed to load reference file")
            return None, None, {}
        
        # Memory cleanup after reference loading
        gc.collect()
        
        # Load inner scan file (with smart downsampling)
        scan_pcd, scan_analysis = self.load_inner_scan_file()
        
        if scan_pcd is None:
            print("CRITICAL ERROR: Failed to load inner scan file")
            return ref_pcd, None, {'reference': ref_analysis}
        
        # Combined analysis
        combined_analysis = {
            'reference': ref_analysis,
            'inner_scan': scan_analysis,
            'loading_successful': True,
            'ready_for_processing': True
        }
        
        print("\n" + "="*80)
        print("DATA LOADING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Reference: {ref_analysis['num_points']:,} points (preserved exactly)")
        print(f"Inner scan: {scan_analysis['num_points']:,} points (optimized)")
        print("Ready for noise removal and scaling pipeline")
        print("="*80)
        
        return ref_pcd, scan_pcd, combined_analysis
    
    def cleanup_memory(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        print("Memory cleanup performed")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA LOADER (PURE LOADING ONLY)")
    print("=" * 60)
    
    loader = PointCloudLoader()
    
    # Test loading both files
    ref_pcd, scan_pcd, analysis = loader.load_both_files()
    
    if ref_pcd is not None and scan_pcd is not None:
        print("SUCCESS: Data loading test completed")
        print("Both files loaded successfully")
        print("Ready for noise removal pipeline")
    else:
        print("ERROR: Data loading test failed")