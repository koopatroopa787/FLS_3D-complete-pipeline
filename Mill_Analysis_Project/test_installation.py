"""
Quick test of installed libraries - Windows compatible
"""
try:
    import numpy as np
    print("[OK] NumPy imported successfully")
    
    import scipy
    print("[OK] SciPy imported successfully")
    
    import matplotlib.pyplot as plt
    print("[OK] Matplotlib imported successfully")
    
    import trimesh
    print("[OK] Trimesh imported successfully")
    
    import pymeshlab
    print("[OK] PyMeshLab imported successfully")
    
    import pandas as pd
    print("[OK] Pandas imported successfully")
    
    import sklearn
    print("[OK] Scikit-learn imported successfully")
    
    print("\n[SUCCESS] ALL LIBRARIES INSTALLED AND WORKING!")
    print("Environment setup complete for Mill Analysis Project")
    print("\nYou can now proceed with:")
    print("1. Copy data files to data/reference/ and data/inner_scans/")
    print("2. Run the mill analysis pipeline")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
except Exception as e:
    print(f"[ERROR] Error: {e}")
