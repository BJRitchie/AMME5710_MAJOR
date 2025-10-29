import pycolmap


# Summary 
# New PPF matching function (relaxed/less strict) 
# Take inverse trnasformation of PPF result



# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 

# Convert the video into images 
# vid_path = "images/batmo.mp4"
# vid_path = 'images/ben.mp4'
# store_path="images/ben"
# vid_path = 'images/batmo.mp4'
store_path="images/batmo"
# gen_images_from_vid( vid_path, store_path ) 

# Storage files 
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"
sat_model_path = "sat_model"

# Settings 
sift_ops = pycolmap.SiftExtractionOptions()
sift_ops.use_gpu = False # CPU only 
sift_ops.first_octave = 0
sift_ops.num_octaves = 4

# Initialise the pipeline 
sfm_pipeline = pipeline.StrcFromMotion ( 
    db_path, im_path, sparse_path, dense_path, sat_model_path,
    cam_mode    =pycolmap.CameraMode.AUTO, 
    cam_model   ="SIMPLE_RADIAL",  
    reader_ops  =pycolmap.ImageReaderOptions(), 
    sift_ops    =sift_ops, 
    device      =pycolmap.Device.cpu 
) 

# sfm_pipeline.make_reference_ply()
# sfm_pipeline.plot_reference_model()

# sfm_pipeline.resize_ims( store_path, 1200, 10 )
# sfm_pipeline.prep_pointcloud() 
# sfm_pipeline.make_pointcloud()
# sfm_pipeline.clean_pointcloud() 
# sfm_pipeline.plot_pointcloud()

import os
import numpy as np
import open3d as o3d
import copy

def test_ppf_verification_with_satellite_vis(pipeline_obj, save_path="sat_model", voxel_size=0.02):
    """
    Generate synthetic satellite point clouds, save them, test the PPF verification pipeline,
    and visualize the reference, target, and aligned point clouds.
    
    Args:
        pipeline_obj: Your pipeline object with verify_ppf_with_synthetic_data().
        save_path: Path to save synthetic point cloud and ground truth transform.
        voxel_size: Voxel size for PPF / ICP matching in the pipeline.
    """
    os.makedirs(save_path, exist_ok=True)

    # --- Generate synthetic satellite point cloud ---
    print("=== Generating Synthetic Satellite Test Data ===")
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
    mesh.compute_vertex_normals()
    ref_pcd = mesh.sample_points_poisson_disk(15000)

    # Apply known rotation + translation
    angle_deg = np.array([5.0, 5.0, 5.0])
    angle_rad = np.deg2rad(angle_deg)
    R = ref_pcd.get_rotation_matrix_from_xyz(angle_rad)
    t = np.array([0.1, 0.2, 0.05])
    gt_transform = np.eye(4)
    gt_transform[:3, :3] = R
    gt_transform[:3, 3] = t

    target_pcd = copy.deepcopy(ref_pcd).transform(gt_transform)

    # Save reference/target points (pipeline expects one file)
    pcd_save_path = os.path.join(save_path, "points.ply")
    o3d.io.write_point_cloud(pcd_save_path, ref_pcd)
    print(f"Saved synthetic point cloud to: {pcd_save_path}")

    # Save ground truth transform
    gt_file_path = os.path.join(save_path, "synthetic_ground_truth.txt")
    with open(gt_file_path, 'w') as f:
        f.write("Ground Truth Transformation Matrix:\n")
        np.savetxt(f, gt_transform)
    print(f"Ground truth transformation saved to: {gt_file_path}")

    # --- Run verification ---
    print("\n=== Running PPF Verification Pipeline ===")
    success = pipeline_obj.verify_ppf_with_synthetic_data(expected_accuracy_threshold=0.01)

    # --- Visualization ---
    print("\n=== Visualizing Point Clouds ===")
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])   # green (reference)

    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1]) # red (transformed)

    # Apply estimated transform from pipeline
    result = pipeline_obj.surface_matching_ppf_icp(store_path="synthetic", voxel_size=voxel_size)
    if result is not None:
        aligned_vis = copy.deepcopy(target_pcd).transform(result['combined_transform'])
        aligned_vis.paint_uniform_color([0.1, 0.1, 0.8])  # blue (aligned)
        print("Aligned point cloud shown in blue")
        o3d.visualization.draw_geometries([ref_vis, target_vis, aligned_vis],
                                          window_name="PPF Verification Visualization",
                                          width=900, height=700)
    else:
        print("No aligned point cloud available for visualization; showing reference vs target only")
        o3d.visualization.draw_geometries([ref_vis, target_vis],
                                          window_name="Reference vs Target",
                                          width=900, height=700)

    if success:
        print("✅ Test PASSED: PPF pipeline correctly aligned the synthetic satellite!")
    else:
        print("❌ Test FAILED: PPF pipeline did not correctly align the synthetic satellite.")


test_ppf_verification_with_satellite_vis(sfm_pipeline)





















# import open3d as o3d
# import numpy as np
# import copy

# def generate_synthetic_satellite():
#     """
#     Generate a satellite-like point cloud with planar panels
#     """
#     # Create box panels to simulate satellite body
#     body = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
#     panel1 = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.02, depth=0.6)
#     panel1.translate((0.0, 0.3, 0.0))
#     panel2 = copy.deepcopy(panel1).translate((0.8, 0.3, 0.0))
    
#     mesh = body + panel1 + panel2
#     mesh.compute_vertex_normals()
    
#     # Sample points densely
#     ref_pcd = mesh.sample_points_poisson_disk(15000)
    
#     # Apply small rotation + translation
#     angle = np.deg2rad([5.0, 5.0, 5.0])
#     R = ref_pcd.get_rotation_matrix_from_xyz(angle)
#     t = np.array([0.1, 0.2, 0.05])
#     T_gt = np.eye(4)
#     T_gt[:3,:3] = R
#     T_gt[:3,3] = t
    
#     target_pcd = copy.deepcopy(ref_pcd).transform(T_gt)
    
#     # Add small noise
#     pts = np.asarray(target_pcd.points)
#     pts += np.random.normal(scale=0.001, size=pts.shape)
#     target_pcd.points = o3d.utility.Vector3dVector(pts)
    
#     return ref_pcd, target_pcd, T_gt

# def preprocess_pcd(pcd, voxel_size=0.02):
#     """
#     Preprocess point cloud: downsample, estimate normals, extract keypoints
#     """
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=50))
    
#     # Optional: keypoints
#     keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_down)
#     return keypoints
#     # return pcd_down

# def ppf_ransac_align(ref_pcd, target_pcd, voxel_size=0.02):
#     """
#     Perform FPFH + RANSAC initial alignment with relaxed parameters
#     """
#     # Preprocess
#     ref_down = preprocess_pcd(ref_pcd, voxel_size)
#     tgt_down = preprocess_pcd(target_pcd, voxel_size)
    
#     # Compute FPFH
#     radius_feature = voxel_size * 10
#     ref_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         ref_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
#     tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
    
#     # RANSAC
#     distance_threshold = voxel_size * 1.5
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         ref_down, tgt_down, ref_fpfh, tgt_fpfh, mutual_filter=False,
#         max_correspondence_distance=distance_threshold,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n=4,
#         checkers=[
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
#         ],
#         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.999)
#     )
    
#     if result.fitness < 0.05:
#         print("Warning: RANSAC failed, using identity.")
#         return np.eye(4)
    
#     return result.transformation

# def refine_icp(ref_pcd, target_pcd, T_init, voxel_size=0.02):
#     """
#     Refine alignment with Point-to-Plane ICP
#     """
#     distance_threshold = voxel_size * 1.5
#     result_icp = o3d.pipelines.registration.registration_icp(
#         target_pcd, ref_pcd, distance_threshold, T_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     )
#     return result_icp.transformation

# def visualize_alignment(ref_pcd, target_pcd, T_est):
#     ref_vis = copy.deepcopy(ref_pcd)
#     ref_vis.paint_uniform_color([0.1,0.8,0.1])
#     tgt_vis = copy.deepcopy(target_pcd).transform(T_est)
#     tgt_vis.paint_uniform_color([0.8,0.1,0.1])
    
#     o3d.visualization.draw_geometries([ref_vis, tgt_vis])

# # --- MAIN PIPELINE ---
# ref_pcd, target_pcd, T_gt = generate_synthetic_satellite()
# T_ransac = ppf_ransac_align(ref_pcd, target_pcd, voxel_size=0.02)
# T_final = refine_icp(ref_pcd, target_pcd, T_ransac, voxel_size=0.02)

# print("Ground truth:\n", T_gt)
# print("Estimated (RANSAC):\n", T_ransac)
# print("Estimated (RANSAC + ICP):\n", T_final)

# visualize_alignment(ref_pcd, target_pcd, T_final)
