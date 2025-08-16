# Mill Analysis Project Setup Guide

## System Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 or newer
- Git (optional, for version control)
- 8GB+ RAM recommended
- 5GB+ free disk space

## 1. Install Python

### Windows
```bash
# Download Python 3.11+ from python.org
# During installation, check "Add Python to PATH"
# Verify installation:
python --version
pip --version
```

### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify installation
python3 --version
pip3 --version
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv
python3.11 --version
pip3 --version
```

## 2. Create Project Directory

```bash
# Navigate to your desired location
cd C:\  # Windows
# cd ~/  # macOS/Linux

# Create project directory
mkdir Mill_Analysis_Project
cd Mill_Analysis_Project
```

## 3. Create Virtual Environment

```bash
# Create virtual environment
python -m venv mill_analysis_env

# Activate virtual environment
# Windows:
mill_analysis_env\Scripts\activate

# macOS/Linux:
source mill_analysis_env/bin/activate

# Verify activation (you should see (mill_analysis_env) in prompt)
```

## 4. Create Directory Structure

```bash
# Create all required directories
mkdir data
mkdir data\reference
mkdir data\inner_scans
mkdir src
mkdir output
mkdir output\visualizations
mkdir output\aligned_scans
mkdir output\heatmaps
mkdir output\intermediate

# Windows alternative (if mkdir doesn't work with backslashes):
md data
md data\reference
md data\inner_scans
md src
md output
md output\visualizations
md output\aligned_scans
md output\heatmaps
md output\intermediate
```

## 5. Install Dependencies

```bash
# Make sure virtual environment is activated
# Create requirements.txt file first:
```

Create `requirements.txt` file with this content:
```
open3d>=0.19.0
numpy>=1.24.0
matplotlib>=3.7.0
typing-extensions>=4.7.0
scipy>=1.9.0
plotly>=5.14.0
```

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

## 6. Add Source Code Files

Place these files in the `src/` directory:

- `config.py`
- `data_loader.py`
- `noise_removal.py`
- `scaling.py`
- `alignment.py`
- `visualization.py`
- `main_processor.py`
- `heatmap_generator.py`
- `run_pipeline.py`
- `run_heatmap_analysis.py`

## 7. Add Data Files

### Reference File
```bash
# Place your reference file in data/reference/
# File: Mill body - vale verde - v1 (1).PLY
```

### Inner Scan Files
```bash
# Place your scan files in data/inner_scans/
# Files: 8_8-Aug-2024.ply, 3_18-Jan-2024.ply
```

## 8. Verify Directory Structure

Your final structure should look like:
```
Mill_Analysis_Project/
├── mill_analysis_env/          # Virtual environment
├── data/
│   ├── reference/
│   │   └── Mill body - vale verde - v1 (1).PLY
│   └── inner_scans/
│       ├── 8_8-Aug-2024.ply
│       └── 3_18-Jan-2024.ply
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── noise_removal.py
│   ├── scaling.py
│   ├── alignment.py
│   ├── visualization.py
│   ├── main_processor.py
│   ├── heatmap_generator.py
│   ├── run_pipeline.py
│   └── run_heatmap_analysis.py
├── output/
│   ├── visualizations/
│   ├── aligned_scans/
│   ├── heatmaps/
│   └── intermediate/
└── requirements.txt
```

## 9. Test Installation

```bash
# Navigate to src directory
cd src

# Test configuration
python config.py

# Test individual modules
python data_loader.py
python noise_removal.py
python scaling.py
python alignment.py
python visualization.py
python heatmap_generator.py
```

## 10. Run Pipeline

### Standard Pipeline
```bash
cd src
python run_pipeline.py
```

### Heatmap Analysis Only
```bash
cd src
python run_heatmap_analysis.py
```

### Test Specific Examples

```bash
# Run just Example 1 (Standard)
# Edit run_pipeline.py to comment out other examples, or run interactively:
python -c "from run_pipeline import run_mill_analysis; run_mill_analysis()"

# Run just Example 2 (With Heatmap)
python -c "from run_pipeline import run_mill_analysis; run_mill_analysis(include_heatmap=True)"

# Run just Example 3 (Two-scan comparison)
python -c "from run_pipeline import run_two_scan_heatmap_comparison; run_two_scan_heatmap_comparison()"
```

## 11. Troubleshooting

### Virtual Environment Issues
```bash
# Deactivate and recreate if needed
deactivate
rmdir mill_analysis_env /s  # Windows
rm -rf mill_analysis_env    # macOS/Linux

# Recreate
python -m venv mill_analysis_env
# Reactivate and reinstall dependencies
```

### Path Issues
```bash
# Windows: Use raw strings or forward slashes in file paths
# Check that all file paths exist
dir data\reference          # Windows
ls data/reference          # macOS/Linux
```

### Memory Issues
```bash
# If running out of memory, reduce processing limits in config.py
# Edit INNER_SCAN_MAX_POINTS to a smaller value like 1,000,000
```

### Dependency Issues
```bash
# Update pip first
python -m pip install --upgrade pip

# Reinstall specific packages if needed
pip uninstall open3d
pip install open3d

# Clear pip cache if needed
pip cache purge
```

## 12. Expected Output

### Successful Run Indicators
- Console output showing processing steps
- PNG files in `output/visualizations/`
- PLY files in `output/aligned_scans/`
- Heatmap files in `output/heatmaps/`
- No Python errors or crashes

### First Run Expected Time
- Standard pipeline: 2-5 minutes
- With heatmap: 3-7 minutes
- Two-scan comparison: 5-10 minutes

## 13. Daily Usage

```bash
# Always activate environment first
cd Mill_Analysis_Project
mill_analysis_env\Scripts\activate  # Windows
source mill_analysis_env/bin/activate  # macOS/Linux

# Run pipeline
cd src
python run_pipeline.py

# Deactivate when done
deactivate
```

## Support

If setup fails:
1. Check Python version: `python --version`
2. Check virtual environment activation
3. Verify all files are in correct directories
4. Check console output for specific error messages
5. Ensure data files exist and are readable