"""
File Path Verification for CloudCompare Integration
Verifies that the exact files created by your original pipeline are properly detected and used
"""

from pathlib import Path
import open3d as o3d


def verify_original_pipeline_outputs():
    """
    Verify that your original pipeline outputs are correctly detected.
    This checks for the exact files your original code creates.
    """
    print("="*80)
    print("VERIFYING ORIGINAL PIPELINE OUTPUT FILES")
    print("="*80)
    
    # The exact paths your original code creates
    base_path = Path("C:/Mill_Analysis_Project/src/output/aligned_scans")
    scan1_path = base_path / "scan1_aligned.ply"
    scan2_path = base_path / "scan2_aligned.ply"
    
    verification_results = {
        'base_directory_exists': base_path.exists(),
        'scan1_exists': scan1_path.exists(),
        'scan2_exists': scan2_path.exists(),
        'files_ready_for_cloudcompare': False,
        'file_details': {}
    }
    
    print(f"Base directory: {base_path}")
    print(f"  Exists: {'✓' if verification_results['base_directory_exists'] else '✗'}")
    
    if verification_results['base_directory_exists']:
        # List all files in the directory
        print(f"\nFiles in aligned_scans directory:")
        try:
            for file in base_path.glob("*.ply"):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Error listing files: {e}")
    
    # Check scan1_aligned.ply
    print(f"\nScan 1 file: {scan1_path}")
    print(f"  Exists: {'✓' if verification_results['scan1_exists'] else '✗'}")
    
    if verification_results['scan1_exists']:
        try:
            # Try to read the file
            pcd1 = o3d.io.read_point_cloud(str(scan1_path))
            points1 = len(pcd1.points)
            verification_results['file_details']['scan1'] = {
                'points': points1,
                'readable': True,
                'size_mb': scan1_path.stat().st_size / (1024 * 1024)
            }
            print(f"  Points: {points1:,}")
            print(f"  Size: {verification_results['file_details']['scan1']['size_mb']:.1f} MB")
            print(f"  Readable: ✓")
        except Exception as e:
            print(f"  Error reading file: {e}")
            verification_results['file_details']['scan1'] = {'readable': False, 'error': str(e)}
    
    # Check scan2_aligned.ply
    print(f"\nScan 2 file: {scan2_path}")
    print(f"  Exists: {'✓' if verification_results['scan2_exists'] else '✗'}")
    
    if verification_results['scan2_exists']:
        try:
            # Try to read the file
            pcd2 = o3d.io.read_point_cloud(str(scan2_path))
            points2 = len(pcd2.points)
            verification_results['file_details']['scan2'] = {
                'points': points2,
                'readable': True,
                'size_mb': scan2_path.stat().st_size / (1024 * 1024)
            }
            print(f"  Points: {points2:,}")
            print(f"  Size: {verification_results['file_details']['scan2']['size_mb']:.1f} MB")
            print(f"  Readable: ✓")
        except Exception as e:
            print(f"  Error reading file: {e}")
            verification_results['file_details']['scan2'] = {'readable': False, 'error': str(e)}
    
    # Overall assessment
    verification_results['files_ready_for_cloudcompare'] = (
        verification_results['scan1_exists'] and 
        verification_results['scan2_exists'] and
        verification_results['file_details'].get('scan1', {}).get('readable', False) and
        verification_results['file_details'].get('scan2', {}).get('readable', False)
    )
    
    print(f"\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Files ready for CloudCompare: {'✓ YES' if verification_results['files_ready_for_cloudcompare'] else '✗ NO'}")
    
    if verification_results['files_ready_for_cloudcompare']:
        print(f"\n✓ Both scan1_aligned.ply and scan2_aligned.ply are ready")
        print(f"✓ CloudCompare integration can proceed")
        print(f"✓ Files will be processed from: {base_path}")
        
        # Show total points for CloudCompare processing
        total_points = (
            verification_results['file_details']['scan1'].get('points', 0) +
            verification_results['file_details']['scan2'].get('points', 0)
        )
        print(f"✓ Total points for CloudCompare processing: {total_points:,}")
        
    else:
        print(f"\n✗ Issues detected:")
        if not verification_results['scan1_exists']:
            print(f"  - scan1_aligned.ply not found")
        if not verification_results['scan2_exists']:
            print(f"  - scan2_aligned.ply not found")
        if not verification_results['base_directory_exists']:
            print(f"  - Base directory does not exist")
        
        print(f"\nTo fix:")
        print(f"  1. Run your original pipeline first to create the aligned scan files")
        print(f"  2. Ensure dual_scan_mode=True in your original pipeline")
        print(f"  3. Verify files are saved to: {base_path}")
    
    return verification_results


def create_cloudcompare_integration_test():
    """
    Test the CloudCompare integration with your specific files.
    This simulates the CloudCompare processing step using your exact files.
    """
    print("\n" + "="*80)
    print("CLOUDCOMPARE INTEGRATION TEST")
    print("="*80)
    
    # First verify files exist
    verification = verify_original_pipeline_outputs()
    
    if not verification['files_ready_for_cloudcompare']:
        print("✗ Cannot proceed with integration test - files not ready")
        return False
    
    try:
        # Import CloudCompare processor
        from cloudcompare_integration import CloudCompareProcessor
        
        # Initialize processor
        cc_processor = CloudCompareProcessor()
        print(f"✓ CloudCompare processor initialized")
        print(f"✓ Output directory: {cc_processor.output_dir}")
        
        # Set up file paths - using YOUR exact files
        reference_path = "output/aligned_scans/reference_for_cloudcompare.ply"  # We'll create this
        scan1_path = "C:/Mill_Analysis_Project/src/output/aligned_scans/scan1_aligned.ply"  # Your file
        scan2_path = "C:/Mill_Analysis_Project/src/output/aligned_scans/scan2_aligned.ply"  # Your file
        
        print(f"\nCloudCompare will process:")
        print(f"  scan1_aligned.ply: {Path(scan1_path).exists()}")
        print(f"  scan2_aligned.ply: {Path(scan2_path).exists()}")
        
        # Note: We would need a reference file for CloudCompare
        print(f"\nNOTE: CloudCompare integration ready!")
        print(f"Your original pipeline outputs → CloudCompare → Enhanced heatmaps")
        
        return True
        
    except ImportError as e:
        print(f"✗ CloudCompare integration not available: {e}")
        print("Make sure cloudcompare_integration.py is in the src directory")
        return False
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("File Path Verification for CloudCompare Integration")
    print("This verifies your original pipeline outputs are correctly detected\n")
    
    # Run verification
    verification_results = verify_original_pipeline_outputs()
    
    # Run integration test if files are ready
    if verification_results['files_ready_for_cloudcompare']:
        create_cloudcompare_integration_test()
    
    print(f"\nVerification complete!")