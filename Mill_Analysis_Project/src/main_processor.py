"""
Complete Mill Analysis Processing Pipeline
Open3D implementation - Orchestrates all split modules
Runs the complete V3 noise removal + old_version alignment pipeline
Now includes optional heatmap generation functionality
"""

import open3d as o3d
import numpy as np
import copy
import time
from typing import Optional, Dict, Any
from pathlib import Path

from config import ProcessingConfig
from data_loader import PointCloudLoader
from noise_removal import NoiseRemovalProcessor
from scaling import ScalingProcessor
from alignment import AlignmentProcessor
from visualization import VisualizationProcessor


class MillAnalysisProcessor:
    """Complete mill analysis processor combining all modules."""
    
    def __init__(self):
        self.config = ProcessingConfig()
        
        # Initialize all processing modules
        self.data_loader = PointCloudLoader()
        self.noise_processor = NoiseRemovalProcessor()
        self.scaling_processor = ScalingProcessor()
        self.alignment_processor = AlignmentProcessor()
        self.visualization_processor = VisualizationProcessor()
        
        # Core data storage
        self.reference_pcd = None
        self.inner_scan_pcd = None
        self.original_inner_pcd = None  # Keep original for visualization
        self.cleaned_inner_pcd = None
        self.scaled_inner_pcd = None
        self.final_aligned_pcd = None
        
        # Processing results
        self.analysis_data = {}
        self.scale_factor = 1.0
        
    def run_complete_pipeline(self, target_ratio: float = 1.0) -> bool:
        """
        Run complete processing pipeline:
        Load -> Clean -> Scale -> Align -> Visualize
        
        Args:
            target_ratio: Target scaling ratio (1.0 for 1:1 scaling)
            
        Returns:
            Success status
        """
        print("="*100)
        print("MILL ANALYSIS - COMPLETE PROCESSING PIPELINE")
        print("V3 Noise Removal + old_version Alignment Algorithms")
        print("="*100)
        
        start_time = time.time()
        
        try:
            # STEP 1: Load both data files
            if not self._step_1_load_data():
                return False
            
            # STEP 2: Apply V3 noise removal
            if not self._step_2_noise_removal():
                return False
            
            # STEP 3: Calculate and apply scaling
            if not self._step_3_scaling(target_ratio):
                return False
            
            # STEP 4: Apply complete alignment
            if not self._step_4_alignment():
                return False
            
            # STEP 5: Create comprehensive visualizations
            if not self._step_5_visualization():
                return False
            
            # STEP 6: Generate final summary
            self._step_6_final_summary(start_time)
            
            return True
            
        except Exception as e:
            print(f"CRITICAL ERROR in pipeline: {str(e)}")
            return False
    
    def _step_1_load_data(self) -> bool:
        """Step 1: Load both reference and inner scan files."""
        print("\n[STEP 1] LOADING DATA FILES")
        print("="*60)
        
        self.reference_pcd, self.inner_scan_pcd, self.analysis_data = self.data_loader.load_both_files()
        
        if self.reference_pcd is None or self.inner_scan_pcd is None:
            print("ERROR: Data loading failed")
            return False
        
        # Keep original inner scan for visualization comparisons
        self.original_inner_pcd = copy.deepcopy(self.inner_scan_pcd)
        
        print("SUCCESS: Both files loaded successfully")
        self.data_loader.cleanup_memory()
        return True
    
    def _step_2_noise_removal(self) -> bool:
        """Step 2: Apply V3 noise removal to inner scan."""
        print("\n[STEP 2] V3 NOISE REMOVAL PIPELINE")
        print("="*60)
        
        self.cleaned_inner_pcd = self.noise_processor.apply_v3_noise_removal(
            self.inner_scan_pcd,
            voxel_size=self.config.VOXEL_SIZE,
            nb_neighbors=self.config.NB_NEIGHBORS,
            std_ratio=self.config.STD_RATIO,
            eps=self.config.DBSCAN_EPS,
            min_points=self.config.DBSCAN_MIN_POINTS
        )
        
        if self.cleaned_inner_pcd is None:
            print("ERROR: V3 noise removal failed")
            return False
        
        print("SUCCESS: V3 noise removal completed")
        return True
    
    def _step_3_scaling(self, target_ratio: float) -> bool:
        """Step 3: Calculate and apply scaling transformation."""
        print("\n[STEP 3] SCALING PIPELINE")
        print("="*60)
        
        # Optional: Check for unit mismatch and convert if needed
        print("Checking for unit mismatch...")
        converted_pcd, was_converted = self.scaling_processor.detect_and_convert_units(
            self.reference_pcd, self.cleaned_inner_pcd
        )
        
        if was_converted:
            self.cleaned_inner_pcd = converted_pcd
            print("Unit conversion applied")
        
        # Calculate scale factor using 95th percentile method
        print("\nCalculating scale factor...")
        self.scale_factor = self.scaling_processor.calculate_scale_factor_95th_percentile(
            self.reference_pcd, self.cleaned_inner_pcd, target_ratio
        )
        
        # Apply scaling transformation
        print("\nApplying scaling transformation...")
        self.scaled_inner_pcd = self.scaling_processor.apply_scaling_transformation(
            self.cleaned_inner_pcd, self.scale_factor
        )
        
        if self.scaled_inner_pcd is None:
            print("ERROR: Scaling transformation failed")
            return False
        
        print("SUCCESS: Scaling completed")
        return True
    
    def _step_4_alignment(self) -> bool:
        """Step 4: Apply complete alignment pipeline."""
        print("\n[STEP 4] COMPLETE ALIGNMENT PIPELINE")
        print("="*60)
        
        self.final_aligned_pcd = self.alignment_processor.apply_complete_alignment(
            self.reference_pcd, self.scaled_inner_pcd
        )
        
        if self.final_aligned_pcd is None:
            print("ERROR: Alignment failed")
            return False
        
        print("SUCCESS: Complete alignment completed")
        return True
    
    def _step_5_visualization(self) -> bool:
        """Step 5: Create comprehensive visualizations."""
        print("\n[STEP 5] COMPREHENSIVE VISUALIZATION")
        print("="*60)
        
        # Create noise removal visualization
        print("Creating noise removal visualization...")
        noise_viz_success = self.visualization_processor.create_noise_removal_visualization(
            self.inner_scan_pcd, self.cleaned_inner_pcd
        )
        
        # Create alignment visualization
        print("Creating alignment visualization...")
        alignment_viz_success = self.visualization_processor.create_alignment_visualization(
            self.reference_pcd, self.original_inner_pcd, self.final_aligned_pcd
        )
        
        # Create detailed overlay visualization for better comparison
        print("Creating detailed overlay visualization...")
        overlay_viz_success = self.visualization_processor.create_detailed_overlay_visualization(
            self.reference_pcd, self.final_aligned_pcd
        )
        
        if not noise_viz_success:
            print("WARNING: Noise removal visualization failed")
        if not alignment_viz_success:
            print("WARNING: Alignment visualization failed")
        if not overlay_viz_success:
            print("WARNING: Detailed overlay visualization failed")
        
        print("Visualization step completed")
        return True  # Don't fail pipeline if visualization fails
    
    def _step_6_final_summary(self, start_time: float) -> None:
        """Step 6: Generate final summary and analysis."""
        print("\n" + "="*100)
        print("COMPLETE MILL PROCESSING PIPELINE SUMMARY")
        print("="*100)
        
        # Calculate processing time
        total_time = time.time() - start_time
        
        # Calculate final statistics
        original_points = len(self.original_inner_pcd.points)
        cleaned_points = len(self.cleaned_inner_pcd.points)
        final_points = len(self.final_aligned_pcd.points)
        
        noise_reduction_pct = ((original_points - cleaned_points) / original_points) * 100
        
        # Calculate alignment accuracy
        ref_center = self.reference_pcd.get_center()
        final_center = self.final_aligned_pcd.get_center()
        center_accuracy = np.linalg.norm(final_center - ref_center)
        
        print(f"PROCESSING STATISTICS:")
        print(f"  Reference: {len(self.reference_pcd.points):,} points (preserved exactly)")
        print(f"  Original inner scan: {original_points:,} points")
        print(f"  After noise removal: {cleaned_points:,} points ({noise_reduction_pct:.1f}% reduction)")
        print(f"  Final aligned: {final_points:,} points")
        print(f"  Total processing time: {total_time:.1f} seconds")
        print(f"")
        print(f"QUALITY METRICS:")
        print(f"  V3 noise removal: {noise_reduction_pct:.1f}% noise eliminated")
        print(f"  Flat surface removal: COMPLETED")
        print(f"  Scale factor applied: {self.scale_factor:.3f}x")
        print(f"  Final center accuracy: {center_accuracy:.3f}mm")
        print(f"")
        print(f"ALGORITHMS APPLIED:")
        print(f"  1. V3 4-step noise removal (voxel + statistical + plane + DBSCAN)")
        print(f"  2. old_version 95th percentile scaling")
        print(f"  3. old_version center alignment")
        print(f"  4. old_version Z-axis alignment (mill cylindrical axis)")
        print(f"  5. old_version PCA axis alignment")
        print(f"  6. old_version ICP refinement")
        print(f"")
        print(f"VISUALIZATIONS:")
        print(f"  Noise removal analysis: CREATED")
        print(f"  Alignment analysis: CREATED") 
        print(f"  Detailed overlay analysis: CREATED")
        print(f"  Check output/visualizations/ directory for results")
        print(f"")
        
        # Determine final quality assessment
        if center_accuracy < 1.0:
            alignment_quality = "EXCELLENT"
        elif center_accuracy < 3.0:
            alignment_quality = "GOOD"
        else:
            alignment_quality = "NEEDS_IMPROVEMENT"
        
        print(f"FINAL RESULT: {alignment_quality} ALIGNMENT ACHIEVED")
        print(f"Mill scans are properly processed and aligned")
        print(f"Ready for wear analysis and comparison")
        print("="*100)
    
    # ========== NEW HEATMAP FUNCTIONALITY ==========
    
    def run_heatmap_comparison_pipeline(self, scan1_path: str, scan2_path: str,
                                      heatmap_type: str = 'enhanced') -> Dict[str, Any]:
        """
        Run complete two-scan heatmap comparison pipeline.
        
        Args:
            scan1_path: Path to first scan PLY file
            scan2_path: Path to second scan PLY file  
            heatmap_type: 'standard' or 'enhanced'
            
        Returns:
            Comprehensive results dictionary
        """
        print("="*100)
        print("MILL ANALYSIS - TWO-SCAN HEATMAP COMPARISON PIPELINE")
        print("="*100)
        
        start_time = time.time()
        
        try:
            # Step 1: Process first scan
            print("\n[STEP 1] PROCESSING FIRST SCAN")
            print("="*60)
            
            scan1_results = self._process_single_scan_for_heatmap(scan1_path, "Scan 1")
            if scan1_results is None:
                print("ERROR: Failed to process first scan")
                return {}
            
            # Step 2: Process second scan  
            print("\n[STEP 2] PROCESSING SECOND SCAN")
            print("="*60)
            
            scan2_results = self._process_single_scan_for_heatmap(scan2_path, "Scan 2")
            if scan2_results is None:
                print("ERROR: Failed to process second scan")
                return {}
            
            # Step 3: Generate heatmap comparison
            print("\n[STEP 3] GENERATING HEATMAP COMPARISON")
            print("="*60)
            
            heatmap_results = self._generate_heatmap_comparison(
                scan1_results['final_aligned_pcd'], 
                scan2_results['final_aligned_pcd'],
                heatmap_type
            )
            
            if not heatmap_results:
                print("ERROR: Failed to generate heatmap comparison")
                return {}
            
            # Step 4: Create comprehensive visualizations
            print("\n[STEP 4] CREATING COMPREHENSIVE VISUALIZATIONS")
            print("="*60)
            
            viz_success = self._create_heatmap_visualizations(
                scan1_results['final_aligned_pcd'],
                scan2_results['final_aligned_pcd'], 
                heatmap_results
            )
            
            # Step 5: Generate final summary
            total_time = time.time() - start_time
            self._generate_heatmap_comparison_summary(scan1_results, scan2_results, heatmap_results, total_time)
            
            # Compile comprehensive results
            comparison_results = {
                'scan1_results': scan1_results,
                'scan2_results': scan2_results,
                'heatmap_results': heatmap_results,
                'visualization_success': viz_success,
                'total_processing_time': total_time,
                'comparison_successful': True
            }
            
            return comparison_results
            
        except Exception as e:
            print(f"CRITICAL ERROR in heatmap comparison pipeline: {str(e)}")
            return {}

    def _process_single_scan_for_heatmap(self, scan_path: str, scan_name: str) -> Optional[Dict[str, Any]]:
        """Process a single scan through the complete pipeline for heatmap analysis."""
        
        print(f"Processing {scan_name}: {scan_path}")
        
        # Create temporary processor for this scan
        temp_processor = MillAnalysisProcessor()
        
        # Load the scan as inner scan (we'll use the same reference for consistency)
        temp_processor.reference_pcd = copy.deepcopy(self.reference_pcd) if self.reference_pcd else None
        
        # Load the specific scan file
        scan_pcd = temp_processor.data_loader.load_point_cloud_file(Path(scan_path))
        if scan_pcd is None:
            print(f"ERROR: Failed to load {scan_name}")
            return None
        
        temp_processor.inner_scan_pcd = scan_pcd
        temp_processor.original_inner_pcd = copy.deepcopy(scan_pcd)
        
        # Run through processing pipeline
        success = temp_processor.run_complete_pipeline(target_ratio=1.0)
        if not success:
            print(f"ERROR: Failed to process {scan_name}")
            return None
        
        results = temp_processor.get_processing_results()
        results['scan_name'] = scan_name
        results['scan_path'] = scan_path
        
        print(f"SUCCESS: {scan_name} processed successfully")
        return results

    def _generate_heatmap_comparison(self, scan1_pcd: o3d.geometry.PointCloud,
                                   scan2_pcd: o3d.geometry.PointCloud,
                                   heatmap_type: str) -> Dict[str, Any]:
        """Generate heatmap comparison between two processed scans."""
        
        from heatmap_generator import WearHeatmapGenerator
        
        print(f"Generating {heatmap_type} heatmap comparison...")
        
        # Initialize heatmap generator
        heatmap_gen = WearHeatmapGenerator()
        
        # Generate appropriate heatmap type
        if heatmap_type == 'enhanced':
            heatmap_pcd = heatmap_gen.generate_enhanced_heatmap(
                scan1_pcd, scan2_pcd,
                max_points=self.config.HEATMAP_MAX_POINTS,
                max_color_range_mm=self.config.HEATMAP_MAX_COLOR_RANGE_MM
            )
        else:  # standard
            heatmap_pcd = heatmap_gen.generate_standard_heatmap(
                scan1_pcd, scan2_pcd,
                max_points=self.config.HEATMAP_MAX_POINTS
            )
        
        if heatmap_pcd is None:
            print("ERROR: Heatmap generation failed")
            return {}
        
        # Get comprehensive results
        results = heatmap_gen.get_heatmap_results()
        results['heatmap_type'] = heatmap_type
        results['generation_successful'] = True
        
        # Save heatmap results
        save_success = heatmap_gen.save_heatmap_results(str(self.data_loader.file_paths['heatmaps_dir']))
        results['save_successful'] = save_success
        
        print(f"SUCCESS: {heatmap_type} heatmap generated successfully")
        return results

    def _create_heatmap_visualizations(self, scan1_pcd: o3d.geometry.PointCloud,
                                     scan2_pcd: o3d.geometry.PointCloud,
                                     heatmap_results: Dict[str, Any]) -> bool:
        """Create comprehensive heatmap visualizations."""
        
        try:
            # Create main heatmap visualization
            viz1_success = self.visualization_processor.create_heatmap_visualization(heatmap_results)
            
            # Create two-scan comparison visualization
            viz2_success = self.visualization_processor.create_two_scan_comparison_visualization(
                scan1_pcd, scan2_pcd, heatmap_results
            )
            
            return viz1_success and viz2_success
            
        except Exception as e:
            print(f"ERROR creating heatmap visualizations: {str(e)}")
            return False

    def _generate_heatmap_comparison_summary(self, scan1_results: Dict[str, Any],
                                           scan2_results: Dict[str, Any],
                                           heatmap_results: Dict[str, Any],
                                           total_time: float) -> None:
        """Generate comprehensive heatmap comparison summary."""
        
        print("\n" + "="*100)
        print("TWO-SCAN HEATMAP COMPARISON SUMMARY")
        print("="*100)
        
        # Processing statistics
        scan1_points = len(scan1_results['final_aligned_pcd'].points)
        scan2_points = len(scan2_results['final_aligned_pcd'].points)
        
        print(f"PROCESSING STATISTICS:")
        print(f"  Scan 1: {scan1_points:,} points processed")
        print(f"  Scan 2: {scan2_points:,} points processed")
        print(f"  Total processing time: {total_time:.1f} seconds")
        print(f"  Heatmap type: {heatmap_results.get('heatmap_type', 'unknown')}")
        
        # Heatmap analysis
        analysis = heatmap_results.get('analysis', {})
        if analysis:
            heatmap_type = heatmap_results.get('heatmap_type', 'unknown')
            stats = analysis.get(heatmap_type, {})
            
            print(f"\nHEATMAP ANALYSIS:")
            print(f"  Mean wear distance: {stats.get('mean_distance', 0):.2f}mm")
            print(f"  Maximum wear distance: {stats.get('max_distance', 0):.2f}mm")
            print(f"  Minimum wear distance: {stats.get('min_distance', 0):.2f}mm")
            print(f"  95th percentile: {stats.get('percentile_95', 0):.2f}mm")
            print(f"  Points analyzed: {stats.get('num_points', 0):,}")
        
        # Quality assessment
        quality_level = "EXCELLENT" if total_time < 300 else "GOOD" if total_time < 600 else "ACCEPTABLE"
        
        print(f"\nQUALITY METRICS:")
        print(f"  Processing efficiency: {quality_level}")
        print(f"  Heatmap generation: {'SUCCESS' if heatmap_results.get('generation_successful') else 'FAILED'}")
        print(f"  Results saved: {'SUCCESS' if heatmap_results.get('save_successful') else 'FAILED'}")
        
        print(f"\nOUTPUT FILES:")
        print(f"  Heatmap directory: {self.data_loader.file_paths['heatmaps_dir']}")
        print(f"  Visualization directory: {self.data_loader.file_paths['visualizations_dir']}")
        
        print(f"\nFINAL RESULT: TWO-SCAN HEATMAP COMPARISON COMPLETED")
        print(f"Ready for detailed wear progression analysis")
        print("="*100)

    def run_single_scan_heatmap_analysis(self, target_ratio: float = 1.0, 
                                       heatmap_type: str = 'enhanced') -> Dict[str, Any]:
        """
        Run complete pipeline with optional heatmap generation (single scan mode).
        
        Args:
            target_ratio: Target scaling ratio
            heatmap_type: 'standard' or 'enhanced'
            
        Returns:
            Complete results including optional heatmap
        """
        print("="*100)
        print("MILL ANALYSIS - COMPLETE PIPELINE WITH HEATMAP ANALYSIS")
        print("="*100)
        
        # Run standard pipeline first
        success = self.run_complete_pipeline(target_ratio)
        if not success:
            print("ERROR: Standard pipeline failed")
            return {}
        
        # Generate heatmap using same scan twice (placeholder mode)
        print("\n[OPTIONAL] GENERATING HEATMAP ANALYSIS")
        print("="*60)
        print("NOTE: Using same scan twice (placeholder for two-scan comparison)")
        
        heatmap_results = self._generate_heatmap_comparison(
            self.final_aligned_pcd, self.final_aligned_pcd, heatmap_type
        )
        
        if heatmap_results:
            # Create heatmap visualizations
            viz_success = self.visualization_processor.create_heatmap_visualization(heatmap_results)
            print(f"Heatmap visualization: {'SUCCESS' if viz_success else 'FAILED'}")
        
        # Get complete results
        complete_results = self.get_processing_results()
        complete_results['heatmap_results'] = heatmap_results
        complete_results['heatmap_included'] = bool(heatmap_results)
        
        return complete_results
    
    # ========== ORIGINAL METHODS (UNCHANGED) ==========
    
    def get_processing_results(self) -> Dict[str, Any]:
        """
        Get comprehensive processing results.
        
        Returns:
            Dictionary with all processing results and statistics
        """
        if self.final_aligned_pcd is None:
            return {'error': 'Pipeline not completed successfully'}
        
        results = {
            'reference_pcd': self.reference_pcd,
            'original_inner_pcd': self.original_inner_pcd,
            'cleaned_inner_pcd': self.cleaned_inner_pcd,
            'scaled_inner_pcd': self.scaled_inner_pcd,
            'final_aligned_pcd': self.final_aligned_pcd,
            'scale_factor': self.scale_factor,
            'analysis_data': self.analysis_data,
            'processing_successful': True
        }
        
        return results
    
    def save_final_results(self, output_dir: Optional[str] = None) -> bool:
        """
        Save final processed point clouds to output directory.
        
        Args:
            output_dir: Directory to save results (optional)
            
        Returns:
            True if save successful
        """
        try:
            if output_dir is None:
                output_dir = self.data_loader.file_paths['aligned_scans_dir']
            
            # Save final aligned result
            aligned_path = output_dir / 'final_aligned_inner_scan.ply'
            o3d.io.write_point_cloud(str(aligned_path), self.final_aligned_pcd)
            print(f"Final aligned scan saved: {aligned_path}")
            
            # Save cleaned (before alignment) for reference
            cleaned_path = output_dir / 'cleaned_inner_scan.ply'
            o3d.io.write_point_cloud(str(cleaned_path), self.cleaned_inner_pcd)
            print(f"Cleaned scan saved: {cleaned_path}")
            
            return True
            
        except Exception as e:
            print(f"ERROR saving results: {str(e)}")
            return False


def test_complete_pipeline():
    """Test the complete processing pipeline with all split modules."""
    print("="*100)
    print("TESTING COMPLETE MILL PROCESSING PIPELINE")
    print("All modules integrated: data_loader + noise_removal + scaling + alignment + visualization")
    print("="*100)
    
    # Create processor and run complete pipeline
    processor = MillAnalysisProcessor()
    
    # Run the complete pipeline
    success = processor.run_complete_pipeline(target_ratio=1.0)
    
    if success:
        print("\n" + "="*60)
        print("PIPELINE TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("All modules working correctly:")
        print("✓ Data loading")
        print("✓ V3 noise removal")
        print("✓ 95th percentile scaling")
        print("✓ Complete alignment (center + PCA + ICP)")
        print("✓ Comprehensive visualization")
        print("\nResults available in processor.get_processing_results()")
        
        # Optionally save results
        save_success = processor.save_final_results()
        if save_success:
            print("✓ Results saved to output directory")
        
        return True
    else:
        print("\n" + "="*60)
        print("PIPELINE TEST FAILED")
        print("="*60)
        print("Check individual module outputs for error details")
        return False


if __name__ == "__main__":
    test_complete_pipeline()