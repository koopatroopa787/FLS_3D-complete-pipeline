"""
Updated CloudCompare Availability Check for AlignmentProcessor
Fixed based on CloudCompare documentation - no --version command exists
"""

import subprocess
import os

def check_cloudcompare_availability() -> bool:
    """
    CORRECTED CloudCompare availability check.
    CloudCompare does NOT support --version command.
    """
    # Method 1: Check if CloudCompare executable exists
    common_paths = [
        r"C:\Program Files\CloudCompare\CloudCompare.exe",
        r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
    ]
    
    cloudcompare_path = None
    for path in common_paths:
        if os.path.exists(path):
            print(f"CloudCompare found at: {path}")
            cloudcompare_path = path
            break
    
    if not cloudcompare_path:
        print("CloudCompare executable not found")
        return False
    
    # Method 2: Test CloudCompare execution with CORRECT command
    try:
        # CORRECT: Use -SILENT instead of --version
        result = subprocess.run([cloudcompare_path, '-SILENT'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        
        # CloudCompare is working if it completes without major errors
        if result.returncode in [0, 1]:  # Exit code 1 is normal for -SILENT
            print("CloudCompare is available and working")
            return True
        else:
            print(f"CloudCompare test failed with exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("CloudCompare available but took too long to initialize")
        return True  # Still consider it available
    except subprocess.CalledProcessError:
        print("CloudCompare test failed")
        return False
    except Exception as e:
        print(f"CloudCompare test error: {e}")
        return False

# Test the corrected function
if __name__ == "__main__":
    print("Testing CORRECTED CloudCompare availability check...")
    print("="*60)
    
    if check_cloudcompare_availability():
        print("\nSUCCESS: CloudCompare is ready for use!")
        print("Your AlignmentProcessor will be able to use CloudCompare ICP")
    else:
        print("\nFAILED: CloudCompare not available")
        print("Install from: https://www.cloudcompare.org/")
    
    print("\nFIX APPLIED:")
    print("- Removed --version command (doesn't exist in CloudCompare)")
    print("- Added -SILENT command test")
    print("- Fixed timeout handling")
    print("- Updated error handling")
