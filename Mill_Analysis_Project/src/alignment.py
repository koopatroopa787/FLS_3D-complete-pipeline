"""
Alignment Algorithms for Mill Analysis Project
Open3D implementation - Extracted from original massive data_loader.py
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
        Moves the inner scan vertically so its surface touches/aligns with reference surface.
        
        Args:
            reference_pcd: Reference point cloud
            center_aligned_pcd: Center-aligned inner scan
            
        Returns:
            Z-axis aligned point cloud (surface positioned)
        """
        print(f"\nApplying Z-axis surface alignment (old_version algorithm)...")
        
        try:
            ref_points = np.asarray(reference_pcd.points)
            aligned_points = np.asarray(center_aligned_pcd.points)
            
            # Find the top surface of reference using 90th percentile (old_version method)
            ref_heights = ref_points[:, 2]
            ref_top_threshold = np.percentile(ref_heights, 90)
            ref_top_mask = ref_heights >= ref_top_threshold
            ref_top_points = ref_points[ref_top_mask]
            
            # Find the surface of the inner scan using 90th percentile
            inner_heights = aligned_points[:, 2]
            inner_surface_threshold = np.percentile(inner_heights, 90)
            inner_surface_mask = inner_heights >= inner_surface_threshold
            inner_surface_points = aligned_points[inner_surface_mask]
            
            print(f"Reference top surface points: {len(ref_top_points):,}")
            print(f"Inner surface points: {len(inner_surface_points):,}")
            
            if len(ref_top_points) > 100 and len(inner_surface_points) > 100:
                # Calculate surface centers for alignment
                ref_top_center = np.mean(ref_top_points, axis=0)
                inner_surface_center = np.mean(inner_surface_points, axis=0)
                
                # Calculate Z-heights for surface positioning
                ref_top_z = np.mean(ref_top_points[:, 2])
                inner_top_z = np.mean(inner_surface_points[:, 2])
                
                print(f"Reference top surface Z: {ref_top_z:.1f}mm")
                print(f"Inner top surface Z: {inner_top_z:.1f}mm")
                
                # Position inner scan so its top surface is slightly below reference top
                # This creates proper mill liner positioning (old_version method)
                height_adjustment = ref_top_z - inner_top_z - 50  # 50mm below reference top
                
                # Create translation vector (only Z-component for surface alignment)
                translation = np.array([0, 0, height_adjustment])
                
                # Apply surface-based positioning
                z_aligned_pcd = copy.deepcopy(center_aligned_pcd)
                z_aligned_pcd = z_aligned_pcd.translate(translation)
                
                # Verify positioning
                final_points = np.asarray(z_aligned_pcd.points)
                final_heights = final_points[:, 2]
                final_top_z = np.percentile(final_heights, 90)
                
                print(f"Surface-based positioning applied:")
                print(f"  Height adjustment: {height_adjustment:.2f}mm")
                print(f"  Reference top surface Z: {ref_top_z:.1f}mm")
                print(f"  Inner surface positioned at Z: {final_top_z:.1f}mm")
                print(f"  Surface separation: {ref_top_z - final_top_z:.1f}mm")
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
        """
        Fallback Z-axis alignment method (from old_version algorithm).
        
        Args:
            reference_pcd: Reference point cloud
            center_aligned_pcd: Center-aligned inner scan
            
        Returns:
            Z-axis aligned point cloud
        """
        print("Applying fallback Z-axis positioning...")
        
        try:
            ref_points = np.asarray(reference_pcd.points)
            ref_center = reference_pcd.get_center()
            scaled_center = center_aligned_pcd.get_center()
            
            # Position at reference center but adjust height to top surface
            ref_top_z = np.percentile(ref_points[:, 2], 90)
            
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
        Apply PCA-based axis alignment (old_version algorithm).
        
        Args:
            reference_pcd: Reference point cloud
            z_axis_aligned_pcd: Z-axis-aligned inner scan
            
        Returns:
            Axis-aligned point cloud
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
        
        ref_axis = ref_pca.components_[0]  # Primary axis
        aligned_axis = aligned_pca.components_[0]  # Primary axis
        
        # Check alignment quality
        axis_alignment = np.abs(np.dot(ref_axis, aligned_axis))
        print(f"Current axis alignment: {axis_alignment:.3f}")
        
        if axis_alignment < 0.95:  # Needs alignment (old_version threshold)
            print("Applying axis rotation...")
            
            # Calculate rotation matrix to align axes
            rotation_matrix = self._calculate_axis_rotation(aligned_axis, ref_axis)
            
            if rotation_matrix is not None:
                # Apply rotation around center
                axis_aligned_pcd = copy.deepcopy(z_axis_aligned_pcd)
                axis_aligned_pcd = axis_aligned_pcd.translate(-aligned_center)
                
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                axis_aligned_pcd = axis_aligned_pcd.transform(transform_matrix)
                
                axis_aligned_pcd = axis_aligned_pcd.translate(aligned_center)
                
                # Verify rotation
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
                print("INFO: Rotation calculation failed, using z-axis alignment")
                return z_axis_aligned_pcd
        else:
            print("INFO: Axes already well aligned")
            return z_axis_aligned_pcd
    
    def _calculate_axis_rotation(self, axis1: np.ndarray, axis2: np.ndarray, 
                                tolerance: float = 0.1) -> Optional[np.ndarray]:
        """Calculate rotation matrix to align two axes (old_version method)."""
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = axis2 / np.linalg.norm(axis2)
        
        dot_product = np.dot(axis1, axis2)
        if abs(abs(dot_product) - 1.0) < tolerance:
            return None  # Already aligned
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(axis1, axis2)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            return None  # Parallel or anti-parallel
            
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
        Apply ICP refinement for final precision alignment (old_version method).
        
        Args:
            reference_pcd: Reference point cloud
            axis_aligned_pcd: Axis-aligned inner scan
            max_iterations: Maximum ICP iterations
            tolerance: Convergence tolerance
            
        Returns:
            ICP-refined point cloud
        """
        print(f"\nApplying ICP refinement (old_version method)...")
        print(f"Max iterations: {max_iterations}, Tolerance: {tolerance}")
        
        try:
            # Apply ICP registration
            icp_result = o3d.pipelines.registration.registration_icp(
                axis_aligned_pcd, reference_pcd,
                max_correspondence_distance=5.0,  # 5mm threshold
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iterations,
                    relative_fitness=tolerance,
                    relative_rmse=tolerance
                )
            )
            
            print(f"ICP converged: {icp_result.converged}")
            print(f"Fitness score: {icp_result.fitness:.4f}")
            print(f"Inlier RMSE: {icp_result.inlier_rmse:.4f}mm")
            
            if icp_result.fitness > 0.3:  # Good alignment threshold
                # Apply ICP transformation
                icp_aligned_pcd = copy.deepcopy(axis_aligned_pcd)
                icp_aligned_pcd.transform(icp_result.transformation)
                
                print("SUCCESS: ICP refinement applied")
                return icp_aligned_pcd
            else:
                print("WARNING: ICP fitness too low, using axis alignment")
                return axis_aligned_pcd
                
        except Exception as e:
            print(f"WARNING: ICP failed ({e}), using axis alignment")
            return axis_aligned_pcd
    
    def apply_complete_alignment(self, reference_pcd: o3d.geometry.PointCloud,
                               scaled_inner_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply complete alignment pipeline: Center → Z-axis → PCA → ICP.
        
        Args:
            reference_pcd: Reference point cloud
            scaled_inner_pcd: Scaled inner scan point cloud
            
        Returns:
            Fully aligned point cloud
        """
        print("\n" + "="*60)
        print("APPLYING COMPLETE ALIGNMENT PIPELINE")
        print("="*60)
        
        # Step 1: Center alignment
        center_aligned = self.apply_center_alignment(reference_pcd, scaled_inner_pcd)
        
        # Step 2: Z-axis alignment (NEW - mill cylindrical axis alignment)
        z_axis_aligned = self.apply_z_axis_alignment(reference_pcd, center_aligned)
        
        # Step 3: PCA axis alignment
        axis_aligned = self.apply_pca_axis_alignment(reference_pcd, z_axis_aligned)
        
        # Step 4: ICP refinement
        final_aligned = self.apply_icp_refinement(reference_pcd, axis_aligned)
        
        # Calculate final alignment accuracy
        ref_center = reference_pcd.get_center()
        final_center = final_aligned.get_center()
        center_accuracy = np.linalg.norm(final_center - ref_center)
        
        print(f"\nCOMPLETE ALIGNMENT SUMMARY:")
        print(f"Final center accuracy: {center_accuracy:.3f}mm")
        
        if center_accuracy < 1.0:
            alignment_quality = "EXCELLENT"
        elif center_accuracy < 3.0:
            alignment_quality = "GOOD"
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
    print("- PCA axis alignment") 
    print("- ICP refinement")
    print("- Complete alignment pipeline")