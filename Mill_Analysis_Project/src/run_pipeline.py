"""
Enhanced Pipeline Runner for Mill Analysis Project
Demonstrates how to use the split architecture with optional heatmap functionality
"""

from main_processor import MillAnalysisProcessor


def run_mill_analysis(include_heatmap: bool = False, heatmap_type: str = 'enhanced'):
    """
    Run the complete mill analysis pipeline with optional heatmap generation.
    
    Args:
        include_heatmap: Whether to include heatmap analysis (default: False for backward compatibility)
        heatmap_type: Type of heatmap ('standard' or 'enhanced')
    
    Returns:
        Processing results dictionary
    """
    print("Starting Mill Analysis Pipeline...")
    print("Using split architecture: data_loader + noise_removal + scaling + alignment + visualization")
    
    if include_heatmap:
        print("+ heatmap_generator (ENABLED)")
    
    # Create the main processor (automatically initializes all modules)
    processor = MillAnalysisProcessor()
    
    # Choose pipeline based on heatmap requirement
    if include_heatmap:
        # Run complete pipeline with heatmap analysis
        print(f"\nRunning ENHANCED pipeline with {heatmap_type} heatmap analysis...")
        results = processor.run_single_scan_heatmap_analysis(
            target_ratio=1.0,
            heatmap_type=heatmap_type
        )
        
        if results.get('processing_successful', False):
            print("\n SUCCESS: Enhanced mill analysis with heatmap completed!")
            print("\nStandard Results:")
            print("- Noise removed and mill structure cleaned")
            print("- Proper scaling applied (1:1 ratio)")
            print("- Reference and inner scan aligned")
            print("- Comprehensive visualizations created")
            
            print("\nHeatmap Results:")
            if results.get('heatmap_included', False):
                heatmap_analysis = results['heatmap_results'].get('analysis', {})
                if heatmap_type in heatmap_analysis:
                    stats = heatmap_analysis[heatmap_type]
                    print(f"- {heatmap_type.title()} heatmap generated (same scan comparison)")
                    print(f"- Mean wear distance: {stats.get('mean_distance', 0):.2f}mm")
                    print(f"- Max wear distance: {stats.get('max_distance', 0):.2f}mm")
                    print(f"- Points analyzed: {stats.get('num_points', 0):,}")
                    
                    if heatmap_type == 'enhanced':
                        print(f"- 11-stage color mapping with side view analysis")
                    else:
                        print(f"- 4-stage color mapping")
            
            print("\nOutput Locations:")
            print("- Standard analysis: output/visualizations/")
            print("- Heatmap analysis: output/heatmaps/")
            
            return results
        else:
            print("\n ERROR: Enhanced mill analysis with heatmap failed")
            return None
    
    else:
        # Run standard pipeline (original functionality)
        print("\nRunning STANDARD pipeline...")
        success = processor.run_complete_pipeline(target_ratio=1.0)
        
        if success:
            print("\n SUCCESS: Standard mill analysis completed!")
            print("\nResults:")
            print("- Noise removed and mill structure cleaned")
            print("- Proper scaling applied (1:1 ratio)")
            print("- Reference and inner scan aligned")
            print("- Comprehensive visualizations created")
            print("- Check output/visualizations/ for analysis plots")
            
            # Get results for further processing if needed
            results = processor.get_processing_results()
            print(f"\nProcessed point clouds available:")
            print(f"- Reference: {len(results['reference_pcd'].points):,} points")
            print(f"- Final aligned: {len(results['final_aligned_pcd'].points):,} points")
            print(f"- Scale factor: {results['scale_factor']:.3f}x")
            
            return results
        else:
            print("\n ERROR: Standard mill analysis failed")
            print("Check console output for detailed error information")
            return None


def run_two_scan_heatmap_comparison(scan1_path: str = None, scan2_path: str = None, 
                                   heatmap_type: str = 'enhanced'):
    """
    Run two-scan heatmap comparison analysis with complete pipeline processing for each scan.
    Each scan goes through: Load -> Noise Removal -> Scaling -> Alignment -> Heatmap Comparison
    
    Args:
        scan1_path: Path to first scan file (Time A)
        scan2_path: Path to second scan file (Time B)
        heatmap_type: Type of heatmap ('standard' or 'enhanced')
    
    Returns:
        Comparison results dictionary
    """
    print("="*80)
    print("TWO-SCAN HEATMAP COMPARISON WITH COMPLETE PIPELINE PROCESSING")
    print("="*80)
    
    processor = MillAnalysisProcessor()
    
    # Use your specific file paths
    if scan1_path is None:
        scan1_path = r"C:\Mill_Analysis_Project\data\inner_scans\8_8-Aug-2024.ply"
    if scan2_path is None:
        scan2_path = r"C:\Mill_Analysis_Project\data\inner_scans\3_18-Jan-2024.ply"
    
    print(f"Processing Configuration:")
    print(f"  Scan 1 (Time A): {scan1_path}")
    print(f"  Scan 2 (Time B): {scan2_path}")
    print(f"  Heatmap Type: {heatmap_type}")
    print(f"  Each scan will go through complete processing pipeline:")
    print(f"    1. Data Loading")
    print(f"    2. V3 Noise Removal")
    print(f"    3. Scaling (95th percentile)")
    print(f"    4. Complete Alignment (Center + PCA + ICP)")
    print(f"    5. Heatmap Generation and Comparison")
    
    # Load reference for consistency
    print("\nLoading reference scan...")
    processor.reference_pcd, _ = processor.data_loader.load_reference_file()
    if processor.reference_pcd is None:
        print("ERROR: Failed to load reference scan")
        return None
    
    print(f"Reference loaded: {len(processor.reference_pcd.points):,} points")
    
    # Run comparison pipeline (this handles complete processing of both scans)
    print("\nStarting two-scan comparison pipeline...")
    results = processor.run_heatmap_comparison_pipeline(scan1_path, scan2_path, heatmap_type)
    
    if results.get('comparison_successful', False):
        print("\n" + "="*80)
        print("SUCCESS: TWO-SCAN HEATMAP COMPARISON COMPLETED")
        print("="*80)
        
        # Display comprehensive results summary
        scan1_results = results.get('scan1_results', {})
        scan2_results = results.get('scan2_results', {})
        heatmap_results = results.get('heatmap_results', {})
        
        print(f"\nPROCESSING SUMMARY:")
        if scan1_results:
            scan1_final = scan1_results.get('final_aligned_pcd')
            if scan1_final:
                print(f"  Scan 1: {len(scan1_final.points):,} points after complete processing")
        
        if scan2_results:
            scan2_final = scan2_results.get('final_aligned_pcd')
            if scan2_final:
                print(f"  Scan 2: {len(scan2_final.points):,} points after complete processing")
        
        print(f"  Processing Time: {results.get('total_processing_time', 0):.1f} seconds")
        
        # Display heatmap analysis results
        analysis = heatmap_results.get('analysis', {})
        if heatmap_type in analysis:
            stats = analysis[heatmap_type]
            print(f"\nHEATMAP ANALYSIS RESULTS ({heatmap_type}):")
            print(f"  Mean wear distance: {stats.get('mean_distance', 0):.2f}mm")
            print(f"  Maximum wear distance: {stats.get('max_distance', 0):.2f}mm")
            print(f"  Minimum wear distance: {stats.get('min_distance', 0):.2f}mm")
            print(f"  95th percentile: {stats.get('percentile_95', 0):.2f}mm")
            print(f"  Points analyzed: {stats.get('num_points', 0):,}")
            
            if heatmap_type == 'enhanced':
                print(f"  Color mapping: 11-stage enhanced with side view analysis")
            else:
                print(f"  Color mapping: 4-stage standard")
        
        print(f"\nOUTPUT LOCATIONS:")
        print(f"  Heatmap files: output/heatmaps/")
        print(f"  Comparison visualizations: output/visualizations/")
        print(f"  Individual scan results: output/aligned_scans/")
        
        print(f"\nWEAR PROGRESSION ANALYSIS:")
        print(f"  Time A (Aug 2024) vs Time B (Jan 2024)")
        print(f"  Heatmap colors show wear differences between time points")
        print(f"  Blue = Low wear, Red = High wear")
        print(f"  Ready for detailed engineering analysis")
        
        return results
    else:
        print("\n" + "="*80)
        print("ERROR: TWO-SCAN HEATMAP COMPARISON FAILED")
        print("="*80)
        print("Check console output above for detailed error information")
        return None


def run_individual_steps_example():
    """
    Example showing how to run individual steps for custom workflows.
    
    This demonstrates the flexibility of the split architecture.
    """
    print("\nDemonstrating individual step control...")
    
    processor = MillAnalysisProcessor()
    
    # Step 1: Load data only
    print("\nStep 1: Loading data...")
    if processor._step_1_load_data():
        print("Data loaded successfully")
        
        # You could inspect or modify data here before continuing
        original_points = len(processor.inner_scan_pcd.points)
        print(f"Loaded {original_points:,} inner scan points")
        
        # Step 2: Noise removal with custom parameters
        print("\nStep 2: Custom noise removal...")
        processor.cleaned_inner_pcd = processor.noise_processor.apply_v3_noise_removal(
            processor.inner_scan_pcd,
            voxel_size=0.01,  # Custom larger voxel size
            nb_neighbors=25,  # Custom neighbor count
            std_ratio=1.5,    # Custom std ratio
            eps=0.08,         # Custom DBSCAN eps
            min_points=150    # Custom min points
        )
        
        if processor.cleaned_inner_pcd is not None:
            cleaned_points = len(processor.cleaned_inner_pcd.points)
            reduction = ((original_points - cleaned_points) / original_points) * 100
            print(f"Custom noise removal: {reduction:.1f}% reduction")
            
            # Could continue with remaining steps using custom cleaned data...
            print("Individual steps completed successfully")
            print("This shows the flexibility of the split architecture")
        else:
            print("Custom noise removal failed")
    else:
        print("Data loading failed")


if __name__ == "__main__":
    print("="*80)
    print("MILL ANALYSIS - ENHANCED PIPELINE RUNNER")
    print("="*80)
    
    # Example 1: Standard pipeline (original functionality - backward compatible)
    #print("\n[EXAMPLE 1] Standard Pipeline (Original)")
    #print("-"*50)
    #standard_results = run_mill_analysis(include_heatmap=False)
    
    # Example 2: Enhanced pipeline with heatmap
    #print("\n[EXAMPLE 2] Enhanced Pipeline with Heatmap")
    #print("-"*50)
    #enhanced_results = run_mill_analysis(include_heatmap=True, heatmap_type='enhanced')
    
    # Example 3: Two-scan comparison with your specific files
    print("\n[EXAMPLE 3] Two-Scan Heatmap Comparison (Aug 2024 vs Jan 2024)")
    print("-"*50)
    comparison_results = run_two_scan_heatmap_comparison(
        scan1_path=r"C:\Mill_Analysis_Project\data\inner_scans\8_8-Aug-2024.ply",
        scan2_path=r"C:\Mill_Analysis_Project\data\inner_scans\3_18-Jan-2024.ply",
        heatmap_type='enhanced'
    )
    
    # Example 4: Individual step control (advanced)
    #print("\n[EXAMPLE 4] Individual Step Control (Advanced)")
    #print("-"*50)
    #run_individual_steps_example()
    
    print("\n" + "="*80)
    print("ENHANCED PIPELINE DEMONSTRATION COMPLETED")
    print("="*80)
    print("\nThe enhanced pipeline provides:")
    print("- Backward compatibility: Standard mode works exactly as before")
    print("- Optional heatmap analysis: Enhanced mode includes wear analysis")
    print("- Two-scan comparison: Compare different time-point scans")
    print("- Individual module access: Custom workflows possible")
    print("- Clean separation of concerns: Each capability is optional")
    print("- Professional code organization: Modular and maintainable")
    
    print("\nUsage Options:")
    print("1. Standard mode: run_mill_analysis()")
    print("2. With heatmap: run_mill_analysis(include_heatmap=True)")
    print("3. Two-scan mode: run_two_scan_heatmap_comparison()")
    print("4. Custom workflow: Use individual processor methods")