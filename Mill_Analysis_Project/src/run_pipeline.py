"""
Integrated Pipeline Runner with CloudCompare Enhancement
Demonstrates the complete integration: Open3D → CloudCompare → Enhanced Heatmaps
Flow: Noise Removal → Open3D Alignment → CloudCompare Processing → C2C Heatmaps
This is the main entry point for the enhanced mill analysis system
"""

from enhanced_main_processor import EnhancedMillAnalysisProcessor
from enhanced_heatmap_generator import EnhancedWearHeatmapGenerator
from datetime import datetime
import os
from pathlib import Path


def run_integrated_mill_analysis_with_cloudcompare(
    dual_scan_mode: bool = True,
    use_file_selector: bool = False,
    use_cloudcompare: bool = True,
    generate_enhanced_heatmap: bool = True
):
    """
    Run the complete integrated mill analysis pipeline with CloudCompare enhancement.
    
    Integration Flow:
    1. Open3D Pipeline: Noise Removal → Scaling → Alignment
    2. CloudCompare Processing: Subsampling → ICP → C2C Distance Computation
    3. Enhanced Heatmap Generation: Using CloudCompare C2C precision data
    
    Args:
        dual_scan_mode: Process two inner scans for comparison (recommended for heatmaps)
        use_file_selector: Use intelligent file selection vs hardcoded paths
        use_cloudcompare: Enable CloudCompare processing for enhanced precision
        generate_enhanced_heatmap: Generate heatmaps using CloudCompare C2C data
        
    Returns:
        Complete analysis results including CloudCompare outputs
    """
    print("="*120)
    print("INTEGRATED MILL ANALYSIS WITH CLOUDCOMPARE ENHANCEMENT")
    print("="*120)
    print("Pipeline Flow:")
    print("  1. Open3D: Noise Removal → Scaling → Alignment")
    print("  2. CloudCompare: Subsampling → ICP → C2C Distance Computation")
    print("  3. Enhanced Heatmaps: Using CloudCompare C2C precision data")
    print("="*120)
    
    # Configuration summary
    print(f"\nConfiguration:")
    print(f"  Dual-scan mode: {dual_scan_mode}")
    print(f"  File selector: {'Intelligent' if use_file_selector else 'Hardcoded paths'}")
    print(f"  CloudCompare: {'ENABLED' if use_cloudcompare else 'DISABLED'}")
    print(f"  Enhanced heatmaps: {'ENABLED' if generate_enhanced_heatmap else 'DISABLED'}")
    
    start_time = datetime.now()
    
    # Initialize the enhanced processor
    processor = EnhancedMillAnalysisProcessor(
        use_file_selector=use_file_selector,
        dual_scan_mode=dual_scan_mode,
        use_cloudcompare=use_cloudcompare
    )
    
    # Run the integrated pipeline
    print(f"\n[MAIN PIPELINE] Starting integrated processing...")
    pipeline_success = processor.run_enhanced_pipeline_with_cloudcompare(target_ratio=1.0)
    
    if not pipeline_success:
        print("ERROR: Integrated pipeline failed")
        return None
    
    # Get CloudCompare results
    cloudcompare_results = processor.get_cloudcompare_results()
    c2c_files = processor.get_c2c_files()
    
    print(f"\n[PIPELINE STATUS] Open3D + CloudCompare processing completed")
    print(f"  CloudCompare successful: {cloudcompare_results is not None}")
    print(f"  C2C files generated: {len(c2c_files)}")
    
    # Generate enhanced heatmaps using CloudCompare C2C data
    heatmap_results = None
    if generate_enhanced_heatmap:
        print(f"\n[ENHANCED HEATMAPS] Generating precision heatmaps...")
        
        heatmap_generator = EnhancedWearHeatmapGenerator()
        
        if c2c_files:
            # Use CloudCompare C2C files for enhanced precision
            print("Using CloudCompare C2C files for enhanced heatmap precision")
            heatmap_results = heatmap_generator.generate_heatmap_from_cloudcompare_c2c(c2c_files)
        else:
            # Fallback to Open3D method
            print("CloudCompare C2C files not available, using Open3D fallback method")
            if dual_scan_mode and processor.inner_scan1_aligned and processor.inner_scan2_aligned:
                heatmap_results = heatmap_generator.generate_enhanced_heatmap(
                    processor.inner_scan1_aligned,
                    processor.inner_scan2_aligned
                )
            else:
                print("WARNING: Insufficient scan data for heatmap generation")
    
    # Compile final results
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    final_results = {
        'processing_successful': pipeline_success,
        'processing_time_seconds': processing_time,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        
        # Open3D results
        'open3d_pipeline': {
            'noise_removal': True,
            'alignment': True,
            'scale_factors': {
                'scan1': processor.scale_factor1,
                'scan2': processor.scale_factor2 if dual_scan_mode else None
            }
        },
        
        # CloudCompare results
        'cloudcompare_processing': {
            'enabled': use_cloudcompare,
            'successful': cloudcompare_results is not None,
            'c2c_files_count': len(c2c_files),
            'c2c_files': [str(f) for f in c2c_files],
            'processing_time': cloudcompare_results.get('processing_time', 0) if cloudcompare_results else 0
        },
        
        # Heatmap results
        'enhanced_heatmaps': {
            'enabled': generate_enhanced_heatmap,
            'successful': heatmap_results and heatmap_results.get('heatmap_generated', False),
            'data_source': heatmap_results.get('data_source', 'None') if heatmap_results else 'None',
            'analysis': heatmap_results.get('analysis', {}) if heatmap_results else {}
        },
        
        # Integration status
        'integration_status': {
            'dual_scan_mode': dual_scan_mode,
            'file_selector_used': use_file_selector,
            'cloudcompare_c2c_available': len(c2c_files) > 0,
            'precision_level': 'CloudCompare C2C' if c2c_files else 'Open3D point-to-point'
        }
    }
    
    # Print comprehensive summary
    _print_integration_summary(final_results)
    
    return final_results


def _print_integration_summary(results: dict):
    """Print comprehensive integration summary."""
    print("\n" + "="*120)
    print("INTEGRATED MILL ANALYSIS - FINAL SUMMARY")
    print("="*120)
    
    # Processing overview
    processing_time = results['processing_time_seconds']
    print(f"\nProcessing Overview:")
    print(f"  Total time: {processing_time:.1f} seconds")
    print(f"  Status: {'SUCCESS' if results['processing_successful'] else 'FAILED'}")
    print(f"  Start: {results['start_time']}")
    print(f"  End: {results['end_time']}")
    
    # Open3D pipeline results
    open3d = results['open3d_pipeline']
    print(f"\nOpen3D Pipeline:")
    print(f"  Noise removal: {'✓' if open3d['noise_removal'] else '✗'}")
    print(f"  Alignment: {'✓' if open3d['alignment'] else '✗'}")
    print(f"  Scan 1 scale factor: {open3d['scale_factors']['scan1']:.6f}")
    if open3d['scale_factors']['scan2']:
        print(f"  Scan 2 scale factor: {open3d['scale_factors']['scan2']:.6f}")
    
    # CloudCompare processing results
    cc = results['cloudcompare_processing']
    print(f"\nCloudCompare Processing:")
    print(f"  Enabled: {'✓' if cc['enabled'] else '✗'}")
    if cc['enabled']:
        print(f"  Status: {'SUCCESS' if cc['successful'] else 'FAILED'}")
        print(f"  Processing time: {cc['processing_time']:.1f} seconds")
        print(f"  C2C files generated: {cc['c2c_files_count']}")
        if cc['c2c_files']:
            print(f"  C2C files:")
            for c2c_file in cc['c2c_files'][:3]:  # Show first 3
                file_path = Path(c2c_file)
                size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
                print(f"    - {file_path.name} ({size_mb:.1f} MB)")
    
    # Enhanced heatmap results
    heatmap = results['enhanced_heatmaps']
    print(f"\nEnhanced Heatmaps:")
    print(f"  Enabled: {'✓' if heatmap['enabled'] else '✗'}")
    if heatmap['enabled']:
        print(f"  Status: {'SUCCESS' if heatmap['successful'] else 'FAILED'}")
        print(f"  Data source: {heatmap['data_source']}")
        
        if heatmap['successful'] and heatmap['analysis']:
            analysis = heatmap['analysis']
            print(f"  Analysis results:")
            print(f"    Points analyzed: {analysis.get('points_analyzed', 0):,}")
            print(f"    Mean distance: {analysis.get('mean_distance', 0):.3f} mm")
            print(f"    Max distance: {analysis.get('max_distance', 0):.3f} mm")
            print(f"    95th percentile: {analysis.get('percentile_95', 0):.3f} mm")
            if 'c2c_file_used' in analysis:
                print(f"    C2C file used: {Path(analysis['c2c_file_used']).name}")
    
    # Integration status
    integration = results['integration_status']
    print(f"\nIntegration Status:")
    print(f"  Dual-scan mode: {'✓' if integration['dual_scan_mode'] else '✗'}")
    print(f"  File selector: {'Intelligent' if integration['file_selector_used'] else 'Hardcoded'}")
    print(f"  CloudCompare C2C available: {'✓' if integration['cloudcompare_c2c_available'] else '✗'}")
    print(f"  Precision level: {integration['precision_level']}")
    
    # Output locations
    print(f"\nOutput Locations:")
    print(f"  Standard visualizations: output/visualizations/")
    print(f"  CloudCompare results: C:/Mill_Analysis_Project/cloudcompare_mill_analysis/")
    print(f"  Enhanced heatmaps: output/heatmaps/")
    print(f"  Aligned scans: output/aligned_scans/")
    
    # Technical details
    print(f"\nTechnical Details:")
    print(f"  Open3D algorithms: V3 noise removal, fixed scaling (1000x), enhanced alignment")
    print(f"  CloudCompare algorithms: Community-proven subsampling, progressive ICP, C2C octree")
    print(f"  Precision: {'CloudCompare octree-optimized C2C' if integration['cloudcompare_c2c_available'] else 'Open3D KDTree point-to-point'}")
    
    print("="*120)


def run_quick_cloudcompare_test():
    """
    Quick test to verify CloudCompare integration is working.
    Uses single scan mode for faster testing.
    """
    print("="*80)
    print("QUICK CLOUDCOMPARE INTEGRATION TEST")
    print("="*80)
    
    results = run_integrated_mill_analysis_with_cloudcompare(
        dual_scan_mode=False,  # Single scan for quick test
        use_file_selector=False,  # Use hardcoded paths
        use_cloudcompare=True,  # Test CloudCompare
        generate_enhanced_heatmap=False  # Skip heatmap for speed
    )
    
    if results:
        cc_successful = results['cloudcompare_processing']['successful']
        c2c_count = results['cloudcompare_processing']['c2c_files_count']
        
        print(f"\n[TEST RESULTS]")
        print(f"  CloudCompare processing: {'SUCCESS' if cc_successful else 'FAILED'}")
        print(f"  C2C files generated: {c2c_count}")
        
        if cc_successful and c2c_count > 0:
            print("✓ CloudCompare integration is working correctly!")
            return True
        else:
            print("✗ CloudCompare integration needs attention")
            return False
    else:
        print("✗ Pipeline failed completely")
        return False


def run_full_precision_analysis():
    """
    Run complete precision analysis with all features enabled.
    This is the recommended workflow for production analysis.
    """
    print("="*80)
    print("FULL PRECISION MILL ANALYSIS")
    print("="*80)
    
    results = run_integrated_mill_analysis_with_cloudcompare(
        dual_scan_mode=True,  # Full dual-scan comparison
        use_file_selector=False,  # Use hardcoded paths (change to True for interactive)
        use_cloudcompare=True,  # Full CloudCompare precision
        generate_enhanced_heatmap=True  # Complete heatmap analysis
    )
    
    return results


# Main execution examples
if __name__ == "__main__":
    print("Integrated Mill Analysis Pipeline with CloudCompare Enhancement")
    print("Choose execution mode:")
    print("1. Quick CloudCompare test (fast, single scan)")
    print("2. Full precision analysis (complete, dual scan)")
    print("3. Custom configuration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nRunning quick CloudCompare integration test...")
        run_quick_cloudcompare_test()
    
    elif choice == "2":
        print("\nRunning full precision analysis...")
        run_full_precision_analysis()
    
    elif choice == "3":
        print("\nCustom configuration:")
        dual_scan = input("Enable dual-scan mode? (y/n): ").lower() == 'y'
        file_selector = input("Use intelligent file selection? (y/n): ").lower() == 'y'
        cloudcompare = input("Enable CloudCompare processing? (y/n): ").lower() == 'y'
        heatmap = input("Generate enhanced heatmaps? (y/n): ").lower() == 'y'
        
        print("\nRunning custom configuration...")
        run_integrated_mill_analysis_with_cloudcompare(
            dual_scan_mode=dual_scan,
            use_file_selector=file_selector,
            use_cloudcompare=cloudcompare,
            generate_enhanced_heatmap=heatmap
        )
    
    else:
        print("Invalid choice. Running full precision analysis by default...")
        run_full_precision_analysis()