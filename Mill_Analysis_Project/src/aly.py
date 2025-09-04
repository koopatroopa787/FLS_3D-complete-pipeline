import numpy as np
from sklearn.neighbors import NearestNeighbors
import struct
import matplotlib.pyplot as plt
import warnings

def detect_ply_format(filepath):
    """Detect if PLY file is ASCII or binary format"""
    try:
        with open(filepath, 'rb') as f:
            header_data = f.read(1000)
            header_text = header_data.decode('utf-8', errors='ignore')
            
            if 'format ascii' in header_text:
                return 'ascii'
            elif 'format binary' in header_text:
                return 'binary'
            else:
                return 'binary'
    except Exception:
        return 'binary'

def parse_ply_header(filepath):
    """Parse PLY header to get vertex count and properties"""
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            try:
                line_str = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                line_str = line.decode('utf-8', errors='ignore').strip()
            
            header_lines.append(line_str)
            if line_str == 'end_header':
                header_end_pos = f.tell()
                break
        
        vertex_count = 0
        has_colors = False
        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if 'property uchar red' in line or 'property uchar r' in line:
                has_colors = True
        
        return vertex_count, has_colors, header_end_pos

def load_ply_binary(filepath):
    """Load binary PLY file with robust error handling"""
    vertex_count, has_colors, header_end_pos = parse_ply_header(filepath)
    
    vertices = []
    colors = []
    
    with open(filepath, 'rb') as f:
        f.seek(header_end_pos)
        
        successful_reads = 0
        for i in range(vertex_count):
            try:
                # Read x, y, z as floats (4 bytes each)
                x = struct.unpack('<f', f.read(4))[0]
                y = struct.unpack('<f', f.read(4))[0]
                z = struct.unpack('<f', f.read(4))[0]
                
                # Check for valid coordinates (not NaN or infinite)
                if np.isfinite([x, y, z]).all():
                    vertices.append([x, y, z])
                    
                    if has_colors:
                        # Read r, g, b as unsigned chars (1 byte each)
                        r = struct.unpack('<B', f.read(1))[0]
                        g = struct.unpack('<B', f.read(1))[0]
                        b = struct.unpack('<B', f.read(1))[0]
                        colors.append([r / 255.0, g / 255.0, b / 255.0])
                    else:
                        colors.append([0.5, 0.5, 0.5])  # Default gray
                    
                    successful_reads += 1
                else:
                    # Skip invalid coordinates
                    if has_colors:
                        f.read(3)  # Skip color bytes
                    
            except struct.error:
                break
            
            # Progress indicator for large files
            if i > 0 and i % 1000000 == 0:
                print(f"  Processed {i:,} vertices ({successful_reads:,} valid)")
    
    print(f"Successfully read {successful_reads:,} valid vertices out of {vertex_count:,} total")
    return np.array(vertices), np.array(colors)

def separate_colors(vertices, colors):
    """Separate red and blue points with improved color detection"""
    # More robust color separation
    red_mask = (colors[:, 0] > 0.3) & (colors[:, 0] > colors[:, 1]) & (colors[:, 0] > colors[:, 2])
    blue_mask = (colors[:, 2] > 0.3) & (colors[:, 2] > colors[:, 0]) & (colors[:, 2] > colors[:, 1])
    
    red_points = vertices[red_mask]
    blue_points = vertices[blue_mask]
    
    return red_points, blue_points, red_mask, blue_mask

def clean_points(points):
    """Remove NaN and infinite values from point cloud"""
    if len(points) == 0:
        return points
    
    # Check for finite values in all dimensions
    finite_mask = np.isfinite(points).all(axis=1)
    clean_pts = points[finite_mask]
    
    removed_count = len(points) - len(clean_pts)
    if removed_count > 0:
        print(f"  Removed {removed_count:,} invalid points (NaN/Inf)")
    
    return clean_pts

def calculate_alignment_metrics_robust(red_points, blue_points, max_sample_size=100000):
    """Calculate alignment metrics with NaN handling and sampling for large datasets"""
    print("Cleaning point data...")
    
    # Clean points (remove NaN/Inf values)
    red_clean = clean_points(red_points)
    blue_clean = clean_points(blue_points)
    
    if len(red_clean) == 0 or len(blue_clean) == 0:
        print("ERROR: No valid points found after cleaning")
        return None, None
    
    print(f"Valid points - Red: {len(red_clean):,}, Blue: {len(blue_clean):,}")
    
    # For very large datasets, use sampling to speed up computation
    if len(red_clean) > max_sample_size:
        print(f"Sampling {max_sample_size:,} red points for faster computation...")
        sample_indices = np.random.choice(len(red_clean), max_sample_size, replace=False)
        red_sample = red_clean[sample_indices]
    else:
        red_sample = red_clean
    
    if len(blue_clean) > max_sample_size:
        print(f"Sampling {max_sample_size:,} blue points for faster computation...")
        sample_indices = np.random.choice(len(blue_clean), max_sample_size, replace=False)
        blue_sample = blue_clean[sample_indices]
    else:
        blue_sample = blue_clean
    
    print("Computing nearest neighbor distances...")
    
    # Find nearest neighbors
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)  # Use all CPU cores
    neigh.fit(blue_sample)
    distances, indices = neigh.kneighbors(red_sample)
    distances = distances.flatten()
    
    # Calculate statistics
    metrics = {
        'max_distance': float(np.max(distances)),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'percentile_95': float(np.percentile(distances, 95)),
        'percentile_75': float(np.percentile(distances, 75)),
        'red_points_count': len(red_clean),
        'blue_points_count': len(blue_clean),
        'sample_size_red': len(red_sample),
        'sample_size_blue': len(blue_sample)
    }
    
    return metrics, distances

def assess_alignment_quality(metrics, tolerance=1.0):
    """Assess alignment quality based on metrics"""
    if metrics is None:
        return "INVALID - No points found"
    
    mean_dist = metrics['mean_distance']
    
    if mean_dist <= tolerance * 0.1:
        quality = "EXCELLENT"
    elif mean_dist <= tolerance * 0.3:
        quality = "GOOD" 
    elif mean_dist <= tolerance * 0.6:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    return quality

def generate_alignment_report(filepath, tolerance=1.0, max_sample_size=100000):
    """Generate comprehensive alignment report with robust error handling"""
    print("=" * 60)
    print("PLY ALIGNMENT ANALYSIS REPORT")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Tolerance: {tolerance}")
    print(f"Max sample size: {max_sample_size:,}")
    print("-" * 60)
    
    try:
        # Load and process data
        format_type = detect_ply_format(filepath)
        print(f"Detected PLY format: {format_type}")
        
        if format_type == 'binary':
            vertices, colors = load_ply_binary(filepath)
        else:
            print("ASCII format not fully supported for this file size. Use binary PLY.")
            return None, None
        
        print(f"Successfully loaded {len(vertices):,} vertices")
        
        red_points, blue_points, red_mask, blue_mask = separate_colors(vertices, colors)
        
        print(f"Color separation results:")
        print(f"  Red points detected: {len(red_points):,}")
        print(f"  Blue points detected: {len(blue_points):,}")
        print(f"  Unclassified points: {len(vertices) - len(red_points) - len(blue_points):,}")
        
        if len(red_points) == 0 and len(blue_points) == 0:
            print("\nWARNING: No colored points detected.")
            return None, None
        
        if len(red_points) == 0 or len(blue_points) == 0:
            print("ERROR: Only one color detected. Need both red and blue points.")
            return None, None
        
        # Calculate alignment with robust handling
        metrics, distances = calculate_alignment_metrics_robust(
            red_points, blue_points, max_sample_size)
        
        if metrics is None:
            return None, None
        
        print()
        print("ALIGNMENT METRICS:")
        print(f"  Maximum Distance: {metrics['max_distance']:.6f}")
        print(f"  Mean Distance: {metrics['mean_distance']:.6f}")
        print(f"  Median Distance: {metrics['median_distance']:.6f}")
        print(f"  Standard Deviation: {metrics['std_distance']:.6f}")
        print(f"  95th Percentile: {metrics['percentile_95']:.6f}")
        print(f"  75th Percentile: {metrics['percentile_75']:.6f}")
        
        if metrics['sample_size_red'] < metrics['red_points_count']:
            print(f"\nNote: Analysis based on {metrics['sample_size_red']:,} sampled red points")
        if metrics['sample_size_blue'] < metrics['blue_points_count']:
            print(f"      and {metrics['sample_size_blue']:,} sampled blue points")
        
        print()
        
        # Quality assessment
        quality = assess_alignment_quality(metrics, tolerance)
        print(f"ALIGNMENT QUALITY: {quality}")
        print()
        
        # Detailed analysis
        outliers = np.sum(distances > tolerance)
        outlier_percentage = (outliers / len(distances)) * 100
        
        print("DETAILED ANALYSIS:")
        print(f"  Points within tolerance ({tolerance}): {len(distances) - outliers:,}")
        print(f"  Points outside tolerance: {outliers:,} ({outlier_percentage:.1f}%)")
        
        if outlier_percentage > 10:
            print("  WARNING: High percentage of misaligned points!")
        elif outlier_percentage > 5:
            print("  CAUTION: Some misalignment detected")
        else:
            print("  GOOD: Most points are well aligned")
        
        print("-" * 60)
        
        return metrics, distances
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def create_distance_histogram(distances, tolerance=1.0, save_path=None):
    """Create histogram with better handling for large datasets"""
    if distances is None or len(distances) == 0:
        print("No distance data to plot")
        return
    
    # Sample distances if too many for plotting
    if len(distances) > 50000:
        sample_distances = np.random.choice(distances, 50000, replace=False)
        print(f"Plotting sample of {len(sample_distances):,} distances")
    else:
        sample_distances = distances
        
    plt.figure(figsize=(12, 8))
    plt.hist(sample_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=tolerance, color='red', linestyle='--', linewidth=2, 
                label=f'Tolerance ({tolerance})')
    plt.axvline(x=np.mean(distances), color='green', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(distances):.3f})')
    plt.axvline(x=np.median(distances), color='orange', linestyle='-', linewidth=2, 
                label=f'Median ({np.median(distances):.3f})')
    
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances Between Red and Blue Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    try:
        plt.show()
    except:
        print("Could not display plot (no display available)")

# Main execution
if __name__ == "__main__":
    ply_file = r"C:\Mill_Analysis_Project\cloudcompare_mill_analysis\verification_overlay.ply"
    tolerance = 1.0  # Adjust based on your requirements
    max_sample_size = 50000  # Reduce for faster computation on large files
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    # Suppress sklearn warnings about large datasets
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print("Starting PLY alignment analysis...")
    print(f"This may take several minutes for large files...")
    print()
    
    # Generate report
    metrics, distances = generate_alignment_report(ply_file, tolerance, max_sample_size)
    
    # Create visualization if data was successfully loaded
    if metrics is not None and distances is not None:
        histogram_path = ply_file.replace('.ply', '_alignment_histogram.png')
        create_distance_histogram(distances, tolerance, histogram_path)
        
        # Save detailed results to file
        output_file = ply_file.replace('.ply', '_alignment_report.txt')
        with open(output_file, 'w') as f:
            f.write(f"PLY Alignment Analysis Report\n")
            f.write(f"File: {ply_file}\n")
            f.write(f"Tolerance: {tolerance}\n")
            f.write(f"Sample Size: {max_sample_size}\n\n")
            f.write(f"Total Red points: {metrics['red_points_count']:,}\n")
            f.write(f"Total Blue points: {metrics['blue_points_count']:,}\n")
            f.write(f"Sample Red points: {metrics['sample_size_red']:,}\n")
            f.write(f"Sample Blue points: {metrics['sample_size_blue']:,}\n\n")
            f.write(f"Maximum Distance: {metrics['max_distance']:.6f}\n")
            f.write(f"Mean Distance: {metrics['mean_distance']:.6f}\n")
            f.write(f"Median Distance: {metrics['median_distance']:.6f}\n")
            f.write(f"Standard Deviation: {metrics['std_distance']:.6f}\n")
            f.write(f"95th Percentile: {metrics['percentile_95']:.6f}\n")
            f.write(f"Quality Assessment: {assess_alignment_quality(metrics, tolerance)}\n")
        
        print(f"Detailed report saved to: {output_file}")
        print()
        print("Analysis completed successfully!")
        
    else:
        print("Analysis failed. Check the error messages above.")
