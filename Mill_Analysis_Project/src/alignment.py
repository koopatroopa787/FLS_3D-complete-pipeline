"""
Alignment Algorithms for Mill Analysis Project
Open3D implementation - Fixed based on reference code
Implements center alignment, PCA axis alignment, Z-axis alignment, and ICP refinement
"""

import open3d as o3d
import numpy as np
import copy
from typing import Optional
from sklearn.decomposition import PCA

from config import ProcessingConfig


class AlignmentProcessor:
    """Alignment processor using proven algorithms from old_version."""
    
    def __init__(self):
        self.config = ProcessingConfig()
    
    def apply_center_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                             scaled_inner_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply center alignment using old_version algorithm.
        
        Args:
            reference_pcd: Reference point cloud
            scaled_inner_pcd: Scaled inner scan point cloud
            
        Returns:
            Center-aligned point cloud
        """
        print(f"\nApplying center alignment (old_version algorithm)...")
        
        ref_points = np.asarray(reference_pcd.points)
        scaled_points = np.asarray(scaled_inner_pcd.points)
        
        # Calculate robust centers (old_version method)
        ref_center = np.mean(ref_points, axis=0)
        scaled_center = np.mean(scaled_points, axis=0)
        
        print(f"Reference center: [{ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f}]")
        print(f"Scaled inner center: [{scaled_center[0]:.2f}, {scaled_center[1]:.2f}, {scaled_center[2]:.2f}]")
        
        # Calculate translation needed
        translation = ref_center - scaled_center
        translation_distance = np.linalg.norm(translation)
        
        print(f"Translation needed: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
        print(f"Translation distance: {translation_distance:.2f}mm")
        
        # Apply center alignment
        center_aligned_pcd = copy.deepcopy(scaled_inner_pcd)
        
        if translation_distance > 0.1:  # Only if significant translation needed
            center_aligned_pcd = center_aligned_pcd.translate(translation)
            print("SUCCESS: Center alignment applied")
        else:
            print("INFO: No significant translation needed")
        
        # Verify alignment
        aligned_center = center_aligned_pcd.get_center()
        print(f"Final center: [{aligned_center[0]:.2f}, {aligned_center[1]:.2f}, {aligned_center[2]:.2f}]")
        
        verification_distance = np.linalg.norm(aligned_center - ref_center)
        print(f"Center alignment accuracy: {verification_distance:.3f}mm")
        
        return center_aligned_pcd
    
    def apply_z_axis_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                              center_aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply Z-axis alignment for mill surface positioning (from old_version algorithm).
        Uses 95th percentile as in the working version.
        """
        print(f"\nApplying Z-axis surface alignment (old_version algorithm)...")
        
        try:
            ref_points = np.asarray(reference_pcd.points)
            aligned_points = np.asarray(center_aligned_pcd.points)
            
            # Find the top surface using 95th percentile (working version method)
            ref_heights = ref_points[:, 2]
            ref_top_threshold = np.percentile(ref_heights, 95)
            ref_top_mask = ref_heights >= ref_top_threshold
            ref_top_points = ref_points[ref_top_mask]
            
            # Find the surface of the inner scan using 95th percentile
            inner_heights = aligned_points[:, 2]
            inner_surface_threshold = np.percentile(inner_heights, 95)
            inner_surface_mask = inner_heights >= inner_surface_threshold
            inner_surface_points = aligned_points[inner_surface_mask]
            
            print(f"Reference top surface points: {len(ref_top_points):,}")
            print(f"Inner surface points: {len(inner_surface_points):,}")
            
            if len(ref_top_points) > 100 and len(inner_surface_points) > 100:
                # Calculate Z-heights for surface positioning
                ref_top_z = np.mean(ref_top_points[:, 2])
                inner_top_z = np.mean(inner_surface_points[:, 2])
                
                print(f"Reference top surface Z: {ref_top_z:.1f}mm")
                print(f"Inner top surface Z: {inner_top_z:.1f}mm")
                
                # Calculate height adjustment (working version method)
                height_adjustment = ref_top_z - inner_top_z
                
                # Create translation vector (only Z-component for surface alignment)
                translation = np.array([0, 0, height_adjustment])
                
                # Apply surface-based positioning
                z_aligned_pcd = copy.deepcopy(center_aligned_pcd)
                z_aligned_pcd = z_aligned_pcd.translate(translation)
                
                # Verify positioning
                final_points = np.asarray(z_aligned_pcd.points)
                final_heights = final_points[:, 2]
                final_top_z = np.percentile(final_heights, 95)
                final_top_mean = np.mean(final_points[final_heights >= final_top_z][:, 2])
                surface_separation = ref_top_z - final_top_mean
                
                print(f"Surface-based positioning applied:")
                print(f"  Height adjustment: {height_adjustment:.2f}mm")
                print(f"  Reference top surface Z: {ref_top_z:.1f}mm")
                print(f"  Inner surface positioned at Z: {final_top_mean:.1f}mm")
                print(f"  Surface separation: {surface_separation:.1f}mm")
                print("SUCCESS: Z-axis surface alignment applied")
                
                return z_aligned_pcd
            else:
                print("Insufficient surface points for positioning, applying fallback...")
                return self._apply_fallback_z_alignment(reference_pcd, center_aligned_pcd)
                
        except Exception as e:
            print(f"Surface positioning failed: {e}")
            return self._apply_fallback_z_alignment(reference_pcd, center_aligned_pcd)
    
    def _apply_fallback_z_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                                   center_aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Fallback Z-axis alignment method."""
        print("Applying fallback Z-axis positioning...")
        
        try:
            ref_points = np.asarray(reference_pcd.points)
            ref_center = reference_pcd.get_center()
            scaled_center = center_aligned_pcd.get_center()
            
            # Position at reference center but adjust height to top surface
            ref_top_z = np.percentile(ref_points[:, 2], 95)
            
            # Calculate translation with height adjustment
            translation = ref_center - scaled_center
            translation[2] = ref_top_z - scaled_center[2] - 100  # Position 100mm below top surface
            
            # Apply fallback positioning
            z_aligned_pcd = copy.deepcopy(center_aligned_pcd)
            z_aligned_pcd = z_aligned_pcd.translate(translation)
            
            print(f"Fallback positioning applied:")
            print(f"  Reference top Z: {ref_top_z:.1f}mm")
            print(f"  Positioned 100mm below reference top surface")
            print("Applied fallback positioning with top surface alignment")
            
            return z_aligned_pcd
            
        except Exception as e:
            print(f"Fallback positioning failed: {e}, using center alignment")
            return center_aligned_pcd
    
    def apply_pca_axis_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                           z_axis_aligned_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply PCA-based axis alignment (FIXED to actually rotate).
        ALWAYS applies meaningful rotation regardless of initial alignment score.
        """
        print(f"\nApplying PCA axis alignment (old_version algorithm)...")
        
        ref_points = np.asarray(reference_pcd.points)
        aligned_points = np.asarray(z_axis_aligned_pcd.points)
        
        ref_center = np.mean(ref_points, axis=0)
        aligned_center = np.mean(aligned_points, axis=0)
        
        # Calculate principal axes using PCA (old_version method)
        ref_pca = PCA(n_components=3)
        aligned_pca = PCA(n_components=3)
        
        ref_pca.fit(ref_points - ref_center)
        aligned_pca.fit(aligned_points - aligned_center)
        
        ref_axis = ref_pca.components_[0]  # PRIMARY axis only
        aligned_axis = aligned_pca.components_[0]  # PRIMARY axis only
        
        # Check alignment quality
        axis_alignment = np.abs(np.dot(ref_axis, aligned_axis))
        print(f"Current axis alignment: {axis_alignment:.3f}")
        
        # ALWAYS apply rotation for consistency - FIXED calculation
        print("Applying axis rotation...")
        
        # Fixed rotation calculation - ensure we actually get a meaningful rotation
        try:
            # Normalize axes
            ref_axis_norm = ref_axis / np.linalg.norm(ref_axis)
            aligned_axis_norm = aligned_axis / np.linalg.norm(aligned_axis)
            
            # Calculate cross product for rotation axis
            rotation_axis = np.cross(aligned_axis_norm, ref_axis_norm)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:  # Not parallel
                # Normalize rotation axis
                rotation_axis = rotation_axis / rotation_axis_norm
                
                # Calculate rotation angle
                dot_product = np.dot(aligned_axis_norm, ref_axis_norm)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                print(f"Rotation angle: {np.degrees(angle):.2f} degrees")
                
                # Create rotation matrix using axis-angle representation
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                # Rodrigues' rotation formula
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                            [rotation_axis[2], 0, -rotation_axis[0]],
                            [-rotation_axis[1], rotation_axis[0], 0]])
                
                rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
                
                # Apply rotation around center
                axis_aligned_pcd = copy.deepcopy(z_axis_aligned_pcd)
                axis_aligned_pcd = axis_aligned_pcd.translate(-aligned_center)
                
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                axis_aligned_pcd = axis_aligned_pcd.transform(transform_matrix)
                
                axis_aligned_pcd = axis_aligned_pcd.translate(aligned_center)
                
                # Verify rotation was applied
                final_points = np.asarray(axis_aligned_pcd.points)
                final_center = np.mean(final_points, axis=0)
                final_pca = PCA(n_components=3)
                final_pca.fit(final_points - final_center)
                final_axis = final_pca.components_[0]
                
                final_alignment = np.abs(np.dot(ref_axis, final_axis))
                print(f"Final axis alignment: {final_alignment:.3f}")
                print("SUCCESS: Axis alignment applied")
                
                return axis_aligned_pcd
                
            else:
                # Axes are parallel or anti-parallel
                dot_product = np.dot(aligned_axis_norm, ref_axis_norm)
                if dot_product < 0:
                    # Anti-parallel - need 180 degree rotation
                    print("Axes are anti-parallel, applying 180-degree rotation")
                    
                    # Find a perpendicular axis for 180-degree rotation
                    if abs(ref_axis_norm[0]) < 0.9:
                        perp_axis = np.array([1, 0, 0])
                    else:
                        perp_axis = np.array([0, 1, 0])
                    
                    # Make it perpendicular to ref_axis
                    perp_axis = perp_axis - np.dot(perp_axis, ref_axis_norm) * ref_axis_norm
                    perp_axis = perp_axis / np.linalg.norm(perp_axis)
                    
                    # 180-degree rotation matrix
                    rotation_matrix = 2 * np.outer(perp_axis, perp_axis) - np.eye(3)
                    
                    # Apply rotation
                    axis_aligned_pcd = copy.deepcopy(z_axis_aligned_pcd)
                    axis_aligned_pcd = axis_aligned_pcd.translate(-aligned_center)
                    
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = rotation_matrix
                    axis_aligned_pcd = axis_aligned_pcd.transform(transform_matrix)
                    
                    axis_aligned_pcd = axis_aligned_pcd.translate(aligned_center)
                    
                    print("SUCCESS: 180-degree axis alignment applied")
                    return axis_aligned_pcd
                else:
                    print("Axes already parallel, no rotation needed")
                    return z_axis_aligned_pcd
            
        except Exception as e:
            print(f"ERROR in rotation calculation: {e}")
            print("Using z-axis alignment without rotation")
            return z_axis_aligned_pcd
    
    def _calculate_axis_rotation(self, axis1: np.ndarray, axis2: np.ndarray, 
                                tolerance: float = 0.1) -> Optional[np.ndarray]:
        """Calculate rotation matrix to align two axes (old_version method)."""
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = axis2 / np.linalg.norm(axis2)
        
        dot_product = np.dot(axis1, axis2)
        if abs(abs(dot_product) - 1.0) < tolerance:
            return np.eye(3)  # Return identity matrix instead of None for consistency
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(axis1, axis2)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            return np.eye(3)  # Return identity matrix for parallel/anti-parallel
            
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0))
        
        # Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        
        rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return rotation_matrix
    
    def apply_icp_refinement(self, reference_pcd: o3d.geometry.PointCloud,
                           axis_aligned_pcd: o3d.geometry.PointCloud,
                           max_iterations: int = 100,
                           tolerance: float = 1e-6) -> o3d.geometry.PointCloud:
        """
        Apply ICP refinement (FIXED for Open3D 0.19.0).
        Fixed parameters and method signature for current Open3D version.
        """
        print(f"\nApplying ICP refinement (FIXED for Open3D 0.19.0)...")
        print(f"Max iterations: {max_iterations}, Tolerance: {tolerance}")
        
        try:
            # Fixed ICP call for Open3D 0.19.0
            threshold = 0.02  # 2cm threshold 
            trans_init = np.identity(4)
            
            reg_p2p = o3d.pipelines.registration.registration_icp(
                axis_aligned_pcd, reference_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )
            
            print(f"ICP fitness score: {reg_p2p.fitness:.4f}")
            print(f"ICP inlier RMSE: {reg_p2p.inlier_rmse:.4f}mm")
            print(f"ICP correspondences: {len(reg_p2p.correspondence_set):,}")
            
            # Always apply ICP transformation for consistency (FIXED)
            icp_aligned_pcd = copy.deepcopy(axis_aligned_pcd)
            icp_aligned_pcd.transform(reg_p2p.transformation)
            
            print("SUCCESS: ICP refinement applied")
            return icp_aligned_pcd
                
        except Exception as e:
            print(f"WARNING: ICP failed ({e}), using axis alignment")
            return axis_aligned_pcd
    
    def apply_complete_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                               scaled_inner_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply complete alignment pipeline: Center → Z-axis → PCA → ICP.
        FIXED to ensure consistent transformations for all scans.
        """
        print("\n" + "="*60)
        print("APPLYING COMPLETE ALIGNMENT PIPELINE")
        print("="*60)
        
        # Step 1: Center alignment
        center_aligned = self.apply_center_alignment(reference_pcd, scaled_inner_pcd)
        
        # Step 2: Z-axis alignment
        z_axis_aligned = self.apply_z_axis_alignment(reference_pcd, center_aligned)
        
        # Step 3: PCA axis alignment (FIXED - always applies rotation)
        axis_aligned = self.apply_pca_axis_alignment(reference_pcd, z_axis_aligned)
        
        # Step 4: ICP refinement (FIXED - always applies transformation)
        final_aligned = self.apply_icp_refinement(reference_pcd, axis_aligned)
        
        # Calculate final alignment accuracy
        ref_center = reference_pcd.get_center()
        final_center = final_aligned.get_center()
        center_accuracy = np.linalg.norm(final_center - ref_center)
        
        print(f"\nCOMPLETE ALIGNMENT SUMMARY:")
        print(f"Final center accuracy: {center_accuracy:.3f}mm")
        
        if center_accuracy < 50:  # 5cm threshold
            alignment_quality = "GOOD"
        elif center_accuracy < 200:  # 20cm threshold
            alignment_quality = "ACCEPTABLE"
        else:
            alignment_quality = "NEEDS_IMPROVEMENT"
        
        print(f"Alignment quality: {alignment_quality}")
        print("="*60)
        
        return final_aligned


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ALIGNMENT MODULE")
    print("=" * 60)
    
    processor = AlignmentProcessor()
    print("Alignment processor created successfully")
    
    print("\nAlignment Module - Ready for pipeline integration")
    print("Algorithms included:")
    print("- Center alignment")
    print("- Z-axis alignment (mill cylindrical axis)")
    print("- PCA axis alignment (primary axis only)")
    print("- ICP refinement")
    print("- Complete alignment pipeline")