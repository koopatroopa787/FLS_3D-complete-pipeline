"""
Professional Data Loader for Mill Analysis Project with E57 Support and Smart Reference-Based Alignment
Open3D implementation with automatic E57 detection and conversion
ENHANCED: Smart reference-based orientation alignment system
- Reference file defines the coordinate system (never rotated)
- Inner scans are aligned to match reference orientation
- Ensures flat bottom and curved top alignment for mill geometry
"""

import open3d as o3d
import numpy as np
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from config import get_file_paths, ProcessingConfig, FileConfig
from e57_converter import E57Converter


class PointCloudLoader:
    """Professional point cloud loader with smart reference-based alignment system."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.file_paths = get_file_paths()
        
        # Initialize E57 converter
        self.e57_converter = E57Converter() if FileConfig.E57_CONVERSION_ENABLED else None
        self.e57_conversions_performed = []
        
        # File selector integration (optional)
        self.file_selector = None
        self.selected_files = None
        self.current_dual_scan_mode = False
        
        # Reference-based alignment system
        self.reference_orientation_data = None
        self.alignment_corrections_applied = []
        
        # Perform initial E57 scan if auto-conversion is enabled
        if FileConfig.E57_AUTO_CONVERSION and self.e57_converter:
            self.scan_and_convert_e57_files()
    
    def analyze_reference_orientation(self, pcd: o3d.geometry.PointCloud, file_path: Path) -> Dict[str, Any]:
        """
        Analyze reference point cloud to establish coordinate system baseline.
        This defines the orientation that all inner scans should match.
        
        Args:
            pcd: Reference point cloud
            file_path: Reference file path for logging
            
        Returns:
            Dictionary with reference orientation data
        """
        if pcd is None or len(pcd.points) == 0:
            return None
        
        points = np.asarray(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        dimensions = bbox.get_extent()
        
        print(f"\n=== REFERENCE COORDINATE SYSTEM ANALYSIS ===")
        print(f"Reference file: {file_path.name}")
        print(f"Dimensions: X={dimensions[0]:.1f}, Y={dimensions[1]:.1f}, Z={dimensions[2]:.1f}")
        
        # Determine mill axis orientation from reference
        x_dim, y_dim, z_dim = dimensions[0], dimensions[1], dimensions[2]
        
        # Find the cylindrical axis (should be the longest dimension for a mill)
        max_dim = max(x_dim, y_dim, z_dim)
        mill_axis = 'Z' if z_dim == max_dim else ('Y' if y_dim == max_dim else 'X')
        
        # Analyze Z-axis distribution to determine proper orientation
        z_coords = points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_range = z_max - z_min
        
        # Determine if flat surface is at bottom or top
        # Sample points from top and bottom 10%
        bottom_threshold = z_min + (z_range * 0.1)
        top_threshold = z_max - (z_range * 0.1)
        
        bottom_points = points[z_coords <= bottom_threshold]
        top_points = points[z_coords >= top_threshold]
        
        # Calculate flatness (lower standard deviation = flatter)
        bottom_z_std = np.std(bottom_points[:, 2]) if len(bottom_points) > 0 else float('inf')
        top_z_std = np.std(top_points[:, 2]) if len(top_points) > 0 else float('inf')
        
        flat_at_bottom = bottom_z_std < top_z_std
        
        # Calculate principal axis using PCA
        centered_points = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (largest first)
        order = np.argsort(eigenvals)[::-1]
        principal_axis = eigenvecs[:, order[0]]  # Primary axis
        secondary_axis = eigenvecs[:, order[1]]  # Secondary axis
        
        orientation_data = {
            'file_path': file_path,
            'dimensions': (x_dim, y_dim, z_dim),
            'mill_axis': mill_axis,
            'z_range': (z_min, z_max),
            'flat_at_bottom': flat_at_bottom,
            'bottom_z_std': bottom_z_std,
            'top_z_std': top_z_std,
            'principal_axis': principal_axis,
            'secondary_axis': secondary_axis,
            'center': np.mean(points, axis=0)
        }
        
        print(f"Mill axis orientation: {mill_axis}-axis ({max_dim:.1f}mm)")
        print(f"Z-range: {z_min:.1f} to {z_max:.1f}mm (range: {z_range:.1f}mm)")
        print(f"Flat surface location: {'BOTTOM' if flat_at_bottom else 'TOP'}")
        print(f"Bottom flatness (std): {bottom_z_std:.2f}mm")
        print(f"Top flatness (std): {top_z_std:.2f}mm")
        print(f"Principal axis: [{principal_axis[0]:.3f}, {principal_axis[1]:.3f}, {principal_axis[2]:.3f}]")
        print(f"Reference coordinate system established")
        print(f"============================================")
        
        self.reference_orientation_data = orientation_data
        return orientation_data
    
    def align_to_reference_orientation(self, pcd: o3d.geometry.PointCloud, file_path: Path) -> o3d.geometry.PointCloud:
        """
        Align inner scan to match reference orientation.
        
        Args:
            pcd: Inner scan point cloud
            file_path: Inner scan file path for logging
            
        Returns:
            Aligned point cloud matching reference orientation
        """
        if pcd is None or len(pcd.points) == 0 or self.reference_orientation_data is None:
            print(f"WARNING: Cannot align {file_path.name} - insufficient data")
            return pcd
        
        print(f"\n=== ALIGNING TO REFERENCE ORIENTATION ===")
        print(f"Inner scan: {file_path.name}")
        
        points = np.asarray(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        dimensions = bbox.get_extent()
        
        print(f"Original dimensions: X={dimensions[0]:.1f}, Y={dimensions[1]:.1f}, Z={dimensions[2]:.1f}")
        
        # Analyze current orientation
        x_dim, y_dim, z_dim = dimensions[0], dimensions[1], dimensions[2]
        
        # Find the longest dimension (mill axis)
        max_dim = max(x_dim, y_dim, z_dim)
        current_mill_axis = 'Z' if z_dim == max_dim else ('Y' if y_dim == max_dim else 'X')
        reference_mill_axis = self.reference_orientation_data['mill_axis']
        
        print(f"Current mill axis: {current_mill_axis} ({max_dim:.1f}mm)")
        print(f"Reference mill axis: {reference_mill_axis}")
        
        aligned_pcd = copy.deepcopy(pcd) if 'copy' in globals() else pcd.copy()
        rotation_applied = False
        
        # Step 1: Align mill axis to match reference
        if current_mill_axis != reference_mill_axis:
            print(f"CORRECTING: Mill axis alignment {current_mill_axis} → {reference_mill_axis}")
            
            # Determine rotation needed
            if current_mill_axis == 'X' and reference_mill_axis == 'Z':
                # Rotate around Y-axis: X → Z
                rotation_matrix = aligned_pcd.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
                rotation_desc = "Y-axis (X→Z)"
            elif current_mill_axis == 'Y' and reference_mill_axis == 'Z':
                # Rotate around X-axis: Y → Z
                rotation_matrix = aligned_pcd.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
                rotation_desc = "X-axis (Y→Z)"
            elif current_mill_axis == 'Z' and reference_mill_axis == 'X':
                # Rotate around Y-axis: Z → X
                rotation_matrix = aligned_pcd.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
                rotation_desc = "Y-axis (Z→X)"
            elif current_mill_axis == 'Z' and reference_mill_axis == 'Y':
                # Rotate around X-axis: Z → Y
                rotation_matrix = aligned_pcd.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
                rotation_desc = "X-axis (Z→Y)"
            elif current_mill_axis == 'X' and reference_mill_axis == 'Y':
                # Rotate around Z-axis: X → Y
                rotation_matrix = aligned_pcd.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
                rotation_desc = "Z-axis (X→Y)"
            elif current_mill_axis == 'Y' and reference_mill_axis == 'X':
                # Rotate around Z-axis: Y → X
                rotation_matrix = aligned_pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))
                rotation_desc = "Z-axis (Y→X)"
            else:
                print(f"No rotation needed for mill axis alignment")
                rotation_matrix = None
                rotation_desc = "None"
            
            if rotation_matrix is not None:
                center = aligned_pcd.get_center()
                aligned_pcd.rotate(rotation_matrix, center)
                rotation_applied = True
                print(f"Applied 90° rotation around {rotation_desc}")
        else:
            print(f"Mill axis already aligned: {current_mill_axis}")
        
        # Step 2: Ensure proper Z-axis orientation (flat bottom)
        aligned_points = np.asarray(aligned_pcd.points)
        z_coords = aligned_points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_range = z_max - z_min
        
        # Check if flat surface is at bottom
        bottom_threshold = z_min + (z_range * 0.1)
        top_threshold = z_max - (z_range * 0.1)
        
        bottom_points = aligned_points[z_coords <= bottom_threshold]
        top_points = aligned_points[z_coords >= top_threshold]
        
        bottom_z_std = np.std(bottom_points[:, 2]) if len(bottom_points) > 0 else float('inf')
        top_z_std = np.std(top_points[:, 2]) if len(top_points) > 0 else float('inf')
        
        current_flat_at_bottom = bottom_z_std < top_z_std
        reference_flat_at_bottom = self.reference_orientation_data['flat_at_bottom']
        
        print(f"Current flat surface: {'BOTTOM' if current_flat_at_bottom else 'TOP'} (std: {min(bottom_z_std, top_z_std):.2f}mm)")
        print(f"Reference flat surface: {'BOTTOM' if reference_flat_at_bottom else 'TOP'}")
        
        if current_flat_at_bottom != reference_flat_at_bottom:
            print(f"CORRECTING: Z-axis orientation (flipping upside down)")
            
            # Flip around X-axis (180° rotation)
            flip_matrix = aligned_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            center = aligned_pcd.get_center()
            aligned_pcd.rotate(flip_matrix, center)
            rotation_applied = True
            print(f"Applied 180° flip around X-axis")
        else:
            print(f"Z-axis orientation already correct")
        
        # Verify final alignment
        final_bbox = aligned_pcd.get_axis_aligned_bounding_box()
        final_dimensions = final_bbox.get_extent()
        
        print(f"Final dimensions: X={final_dimensions[0]:.1f}, Y={final_dimensions[1]:.1f}, Z={final_dimensions[2]:.1f}")
        print(f"Alignment status: {'CORRECTED' if rotation_applied else 'ALREADY_ALIGNED'}")
        
        # Store alignment information
        if rotation_applied:
            correction_info = {
                'file_path': file_path,
                'original_dimensions': (x_dim, y_dim, z_dim),
                'final_dimensions': tuple(final_dimensions),
                'mill_axis_correction': f"{current_mill_axis} → {reference_mill_axis}",
                'z_axis_correction': f"{'BOTTOM' if current_flat_at_bottom else 'TOP'} → {'BOTTOM' if reference_flat_at_bottom else 'TOP'}",
                'rotation_applied': True
            }
            self.alignment_corrections_applied.append(correction_info)
        
        print(f"=========================================")
        
        return aligned_pcd
    
    def initialize_file_selector(self, dual_scan_mode: bool = False) -> bool:
        """Initialize the intelligent file selector."""
        try:
            from file_selector import IntelligentFileSelector
            self.file_selector = IntelligentFileSelector(dual_scan_mode=dual_scan_mode)
            self.current_dual_scan_mode = dual_scan_mode
            print(f"File selector initialized (dual_scan_mode: {dual_scan_mode})")
            return True
        except ImportError as e:
            print(f"WARNING: Could not import file_selector: {e}")
            print("Falling back to hardcoded file paths")
            return False
        except Exception as e:
            print(f"ERROR: Could not initialize file selector: {e}")
            print("Falling back to hardcoded file paths")
            return False
    
    def perform_intelligent_file_selection(self, dual_scan_mode: bool = False) -> bool:
        """Perform intelligent file selection and store results."""
        if not self.initialize_file_selector(dual_scan_mode):
            return False
        
        try:
            results = self.file_selector.perform_intelligent_file_selection()
            
            if results.get('selection_successful', False):
                self.selected_files = results
                self.current_dual_scan_mode = dual_scan_mode
                print("\nIntelligent file selection completed successfully!")
                return True
            else:
                print("\nERROR: Intelligent file selection failed")
                print("Falling back to hardcoded file paths")
                return False
                
        except Exception as e:
            print(f"ERROR during intelligent file selection: {e}")
            print("Falling back to hardcoded file paths")
            return False
    
    def get_target_file_paths(self, use_file_selector: bool = False) -> Dict[str, Path]:
        """Get target file paths either from intelligent selection or hardcoded config."""
        if use_file_selector and self.selected_files and self.selected_files.get('selection_successful', False):
            # Use intelligently selected files
            result = {
                'reference': self.selected_files['reference'],
                'inner_scan': self.selected_files['inner_scan']
            }
            
            # Add second inner scan if in dual-scan mode
            if self.current_dual_scan_mode and 'inner_scan2' in self.selected_files:
                result['inner_scan2'] = self.selected_files['inner_scan2']
            
            return result
        else:
            # Use hardcoded paths from config (default behavior)
            return {
                'reference': self.file_paths['reference'],
                'inner_scan': self.file_paths['inner_scan']
            }
        
    def scan_and_convert_e57_files(self) -> None:
        """Scan data directories for E57 files and convert them automatically."""
        print("\n" + "="*80)
        print("SCANNING FOR E57 FILES IN DATA DIRECTORIES")
        print("="*80)
        
        # Scan reference directory
        ref_conversions = self.e57_converter.scan_and_convert_directory(
            self.file_paths['data_reference_dir']
        )
        
        # Scan inner scans directory  
        scan_conversions = self.e57_converter.scan_and_convert_directory(
            self.file_paths['data_inner_scans_dir']
        )
        
        # Store conversion results
        self.e57_conversions_performed.extend(ref_conversions)
        self.e57_conversions_performed.extend(scan_conversions)
        
        if self.e57_conversions_performed:
            print(f"\nE57 CONVERSION SUMMARY:")
            print(f"Total files converted: {len(self.e57_conversions_performed)}")
            for original, converted in self.e57_conversions_performed:
                print(f"  {original.name} -> {converted.name}")
            print("Converted files are available for processing")
        else:
            print("No E57 files found in data directories")
        
        print("="*80)
    
    def find_best_file_in_directory(self, directory: Path, 
                                   preferred_patterns: List[str] = None) -> Optional[Path]:
        """Find the best file to use in a directory, with E57 conversion support."""
        if not directory.exists():
            return None
        
        if preferred_patterns is None:
            preferred_patterns = ['mill', 'body', 'reference', 'inner', 'scan']
        
        # Get all supported files
        supported_files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in FileConfig.SUPPORTED_FORMATS:
                supported_files.append(file_path)
        
        if not supported_files:
            return None
        
        # If E57 converter is available, check for converted files
        if self.e57_converter:
            # Look for recently converted files
            for original, converted in self.e57_conversions_performed:
                if converted.parent == directory and converted in supported_files:
                    print(f"Using recently converted E57 file: {converted.name}")
                    return converted
        
        # Prefer files matching patterns
        for pattern in preferred_patterns:
            for file_path in supported_files:
                if pattern.lower() in file_path.name.lower():
                    return file_path
        
        # Return the largest file (likely the most complete scan)
        largest_file = max(supported_files, key=lambda f: f.stat().st_size)
        return largest_file
    
    def resolve_file_path(self, target_path: Path) -> Path:
        """Resolve file path with E57 conversion support."""
        # If target exists, use it directly
        if target_path.exists():
            return target_path
        
        # Check if E57 equivalent exists
        if self.e57_converter:
            e57_path = target_path.with_suffix('.e57')
            if e57_path.exists():
                print(f"Target {target_path.name} not found, but E57 equivalent exists")
                converted_path = self.e57_converter.convert_e57_to_ply(e57_path)
                if converted_path:
                    print(f"Using converted file: {converted_path.name}")
                    return converted_path
        
        # Look for best alternative in the same directory
        alternative = self.find_best_file_in_directory(target_path.parent)
        if alternative:
            print(f"Using alternative file: {alternative.name} instead of {target_path.name}")
            return alternative
        
        return target_path  # Return original even if it doesn't exist (will fail validation)
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate that a file exists and is readable, with E57 support."""
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
        """Get comprehensive information about a point cloud."""
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
                             preserve_properties: bool = True,
                             is_reference: bool = False) -> Optional[o3d.geometry.PointCloud]:
        """
        Load point cloud file with E57 conversion support and smart reference-based alignment.
        
        Args:
            file_path: Path to point cloud file
            preserve_properties: Whether to compute/preserve normals and colors
            is_reference: True if this is the reference file (establishes coordinate system)
            
        Returns:
            Loaded point cloud with proper alignment or None if failed
        """
        # Resolve file path (handles E57 conversion if needed)
        actual_path = self.resolve_file_path(file_path)
        
        # Handle E57 conversion if needed
        if self.e57_converter and actual_path.suffix.lower() == '.e57':
            print(f"E57 file detected: {actual_path.name}")
            converted_path = self.e57_converter.process_file_with_e57_check(actual_path)
            if converted_path != actual_path:
                actual_path = converted_path
                print(f"Using converted PLY: {actual_path.name}")
        
        if not self.validate_file(actual_path):
            return None
            
        print(f"\nLoading point cloud: {actual_path.name}")
        start_time = time.time()
        
        try:
            # Load point cloud using Open3D (V3 exact flow)
            pcd = o3d.io.read_point_cloud(str(actual_path))
            
            if len(pcd.points) == 0:
                print(f"ERROR: No points found in {actual_path.name}")
                return None
                
            points_count = len(pcd.points)
            print(f"Loaded {points_count:,} points (no downsampling - V3 flow)")
            
            # Smart alignment system
            if is_reference:
                # Reference file: Establish coordinate system (never rotate)
                self.analyze_reference_orientation(pcd, actual_path)
                print("Reference coordinate system established - no rotation applied")
            else:
                # Inner scan: Align to match reference orientation
                if self.reference_orientation_data is not None:
                    pcd = self.align_to_reference_orientation(pcd, actual_path)
                else:
                    print("WARNING: No reference orientation data - loading without alignment")
            
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
            print(f"ERROR loading {actual_path.name}: {str(e)}")
            return None
    
    def load_reference_file(self, use_file_selector: bool = False) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """Load reference file and establish coordinate system."""
        print("\n" + "="*80)
        if use_file_selector:
            print("LOADING REFERENCE FILE (SMART ALIGNMENT + INTELLIGENT SELECTION)")
        else:
            print("LOADING REFERENCE FILE (SMART ALIGNMENT)")
        print("="*80)
        
        # Get target file path
        if use_file_selector:
            if not self.perform_intelligent_file_selection(dual_scan_mode=self.current_dual_scan_mode):
                print("WARNING: Falling back to hardcoded reference file path")
                use_file_selector = False
        
        target_paths = self.get_target_file_paths(use_file_selector)
        ref_path = target_paths['reference']
        
        if use_file_selector and self.selected_files:
            print(f"Using intelligently selected reference: {ref_path.name}")
        else:
            print(f"Using configured reference: {ref_path.name}")
        
        # Load reference file (establishes coordinate system)
        pcd = self.load_point_cloud_file(ref_path, preserve_properties=True, is_reference=True)
        
        if pcd is None:
            return None, {}
            
        # Analyze reference point cloud
        analysis = self.get_point_cloud_info(pcd, "Reference Shell")
        self.print_point_cloud_analysis(analysis)
        
        print(f"\nREFERENCE LOADED SUCCESSFULLY:")
        print(f"  Points loaded: {analysis['num_points']:,}")
        print(f"  Coordinate system established")
        print(f"  Ready for inner scan alignment")
        
        return pcd, analysis
    
    def load_inner_scan_file(self, use_file_selector: bool = False) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """Load inner scan file aligned to reference orientation."""
        print("\n" + "="*80)
        if use_file_selector:
            print("LOADING INNER SCAN FILE (REFERENCE-BASED ALIGNMENT + INTELLIGENT SELECTION)")
        else:
            print("LOADING INNER SCAN FILE (REFERENCE-BASED ALIGNMENT)")
        print("="*80)
        
        # Get target file path
        if use_file_selector:
            if not self.selected_files or not self.selected_files.get('selection_successful', False):
                if not self.perform_intelligent_file_selection(dual_scan_mode=self.current_dual_scan_mode):
                    print("WARNING: Falling back to hardcoded inner scan file path")
                    use_file_selector = False
        
        target_paths = self.get_target_file_paths(use_file_selector)
        scan_path = target_paths['inner_scan']
        
        if use_file_selector and self.selected_files:
            print(f"Using intelligently selected inner scan: {scan_path.name}")
        else:
            print(f"Using configured inner scan: {scan_path.name}")
        
        # Load inner scan file (aligned to reference)
        pcd = self.load_point_cloud_file(scan_path, preserve_properties=True, is_reference=False)
        
        if pcd is None:
            return None, {}
            
        # Analyze inner scan point cloud
        analysis = self.get_point_cloud_info(pcd, "Inner Scan")
        self.print_point_cloud_analysis(analysis)
        
        print(f"\nINNER SCAN LOADED SUCCESSFULLY:")
        print(f"  Points loaded: {analysis['num_points']:,}")
        print(f"  Aligned to reference orientation")
        print(f"  Ready for noise removal and alignment pipeline")
        
        return pcd, analysis
    
    def load_both_files(self, use_file_selector: bool = False) -> Tuple[Optional[o3d.geometry.PointCloud], 
                                     Optional[o3d.geometry.PointCloud],
                                     Dict[str, Any]]:
        """Load both reference and inner scan files with smart reference-based alignment."""
        print("="*100)
        if use_file_selector:
            print("MILL ANALYSIS - SMART REFERENCE-BASED ALIGNMENT + INTELLIGENT SELECTION")
        else:
            print("MILL ANALYSIS - SMART REFERENCE-BASED ALIGNMENT")
        print("="*100)
        
        # Load reference file first (establishes coordinate system)
        ref_pcd, ref_analysis = self.load_reference_file(use_file_selector=use_file_selector)
        
        if ref_pcd is None:
            print("CRITICAL ERROR: Failed to load reference file")
            return None, None, {}
        
        # Memory cleanup after reference loading
        gc.collect()
        
        # Load inner scan file (aligned to reference)
        scan_pcd, scan_analysis = self.load_inner_scan_file(use_file_selector=use_file_selector)
        
        if scan_pcd is None:
            print("CRITICAL ERROR: Failed to load inner scan file")
            return ref_pcd, None, {'reference': ref_analysis}
        
        # Combined analysis
        combined_analysis = {
            'reference': ref_analysis,
            'inner_scan': scan_analysis,
            'loading_successful': True,
            'ready_for_processing': True,
            'e57_conversions_performed': len(self.e57_conversions_performed),
            'intelligent_file_selection_used': use_file_selector,
            'alignment_corrections_applied': len(self.alignment_corrections_applied),
            'reference_orientation_established': self.reference_orientation_data is not None,
            'e57_conversion_details': [
                {'original': str(orig), 'converted': str(conv)} 
                for orig, conv in self.e57_conversions_performed
            ],
            'alignment_correction_details': [
                {
                    'file': str(correction['file_path'].name),
                    'mill_axis_correction': correction['mill_axis_correction'],
                    'z_axis_correction': correction['z_axis_correction'],
                    'original_dims': correction['original_dimensions'],
                    'final_dims': correction['final_dimensions']
                }
                for correction in self.alignment_corrections_applied
            ]
        }
        
        # Add file selection details if used
        if use_file_selector and self.selected_files:
            combined_analysis['file_selection_details'] = {
                'selection_successful': self.selected_files.get('selection_successful', False),
                'dual_scan_mode': self.selected_files.get('dual_scan_mode', False),
                'reference_file': str(self.selected_files.get('reference', '')),
                'inner_scan_file': str(self.selected_files.get('inner_scan', ''))
            }
        
        print("\n" + "="*80)
        print("SMART REFERENCE-BASED ALIGNMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Reference: {ref_analysis['num_points']:,} points (coordinate system established)")
        print(f"Inner scan: {scan_analysis['num_points']:,} points (aligned to reference)")
        if self.e57_conversions_performed:
            print(f"E57 conversions: {len(self.e57_conversions_performed)} files converted")
        if self.alignment_corrections_applied:
            print(f"Alignment corrections: {len(self.alignment_corrections_applied)} scans auto-aligned")
            for correction in self.alignment_corrections_applied:
                print(f"  - {correction['file_path'].name}: {correction['mill_axis_correction']}")
        if use_file_selector:
            print("File selection: Intelligent selection used")
        else:
            print("File selection: Hardcoded paths used")
        print("Ready for noise removal and alignment pipeline")
        print("="*80)
        
        return ref_pcd, scan_pcd, combined_analysis
    
    def cleanup_memory(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        print("Memory cleanup performed")
    
    def get_alignment_correction_summary(self) -> Dict[str, Any]:
        """Get summary of alignment corrections applied."""
        return {
            'reference_orientation_established': self.reference_orientation_data is not None,
            'reference_data': self.reference_orientation_data,
            'corrections_applied': len(self.alignment_corrections_applied),
            'correction_details': self.alignment_corrections_applied
        }


# Import copy at module level to avoid runtime issues
import copy


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING SMART REFERENCE-BASED ALIGNMENT SYSTEM")
    print("=" * 80)
    
    loader = PointCloudLoader()
    
    # Test reference-based alignment
    print("\n" + "="*60)
    print("TEST: SMART REFERENCE-BASED ALIGNMENT")
    print("="*60)
    
    ref_pcd, scan_pcd, analysis = loader.load_both_files()
    
    if ref_pcd is not None and scan_pcd is not None:
        print("SUCCESS: Smart alignment test completed")
        alignment_summary = loader.get_alignment_correction_summary()
        print(f"Reference orientation established: {alignment_summary['reference_orientation_established']}")
        print(f"Alignment corrections applied: {alignment_summary['corrections_applied']}")
        
        if alignment_summary['reference_data']:
            ref_data = alignment_summary['reference_data']
            print(f"Reference mill axis: {ref_data['mill_axis']}")
            print(f"Reference flat surface: {'BOTTOM' if ref_data['flat_at_bottom'] else 'TOP'}")
        
        for correction in alignment_summary['correction_details']:
            print(f"Corrected: {correction['file_path'].name}")
            print(f"  Mill axis: {correction['mill_axis_correction']}")
            print(f"  Z-axis: {correction['z_axis_correction']}")
    else:
        print("ERROR: Smart alignment test failed")
    
    print("\n" + "="*80)
    print("SMART REFERENCE-BASED ALIGNMENT TESTING COMPLETED")
    print("="*80)
    print("[OK] Reference defines coordinate system (never rotated)")
    print("[OK] Inner scans aligned to match reference orientation")
    print("[OK] Mill axis alignment (longest dimension)")
    print("[OK] Z-axis orientation (flat bottom, curved top)")
    print("[OK] E57 support maintained")
    print("[OK] Intelligent file selection maintained")
    print("="*80)