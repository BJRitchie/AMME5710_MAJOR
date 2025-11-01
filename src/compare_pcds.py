import pycolmap
 
import sfm_pipeline_lib as pipeline 
import gen_synthetic_pcd_lib as gen
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

import numpy as np
import open3d as o3d

# ## GENERATE SFM CLASS ##
# # Convert the video into images 
# store_path= "images/checker_nasa_box"
# vid_path = 'images/checker_nasa_box.mp4'
# gen_images_from_vid( vid_path, store_path ) 


# # Storage files 
# im_path = store_path
# db_path = "database.db"
# sparse_path = "sparse"
# dense_path = "dense"
# sat_model_path = "sat_model"

# # Settings 
# sift_ops = pycolmap.SiftExtractionOptions()
# sift_ops.use_gpu = False # CPU only 
# sift_ops.first_octave = 0
# sift_ops.num_octaves = 4

# # Initialise the pipeline 
# sfm_pipeline = pipeline.StrcFromMotion ( 
#     db_path, im_path, sparse_path, dense_path, sat_model_path,
#     cam_mode    =pycolmap.CameraMode.AUTO, 
#     cam_model   ="SIMPLE_RADIAL",  
#     reader_ops  =pycolmap.ImageReaderOptions(), 
#     sift_ops    =sift_ops, 
#     device      =pycolmap.Device.cpu 
# ) 


## GENERATE REF SYNTHETIC POINT CLOUD ##
# TODO Think these are also computing a target point cloud and gt, but then doing that again the function below - what if GT doesn't match??
# TODO Delete synthetic test data function and add in from here? Or make it consistent 
# Best working cubesat
# ref_pcd = gen.generate_ppf_friendly_cubesat(num_points=20000, noise_std=0.001)

# Basic box 
ref_pcd, _, _ = gen.generate_test_pcds() 


# test_data = gen.generate_data_from_sfm(ref_pcd)
import pickle
with open("sfm_pipeline.pkl", "rb") as f:
    sfm_pipeline = pickle.load(f)

with open("checkerboard.pkl", "rb") as f:
    cb = pickle.load(f)

print("Loaded saved SfM pipeline and checkerboard data")


import point_cloud_utils as pcu

def compute_alignment_errors(ref_pcd, test_pcd):
    """
    Compute Chamfer and Hausdorff distances between two point clouds.

    Args:
        ref_pcd: Open3D PointCloud (reference)
        test_pcd: Open3D PointCloud (aligned test)
    Returns:
        dict with chamfer and hausdorff distances
    """
    ref_pts = np.asarray(ref_pcd.points)
    test_pts = np.asarray(test_pcd.points)

    chamfer = pcu.chamfer_distance(ref_pts, test_pts)
    hausdorff = pcu.hausdorff_distance(ref_pts, test_pts)

    print(f"Chamfer distance: {chamfer:.6f}")
    print(f"Hausdorff distance: {hausdorff:.6f}")

    return chamfer, hausdorff




# Generate synethic test point cloud (mimic sfm)
sfm_pcd = sfm_pipeline.generate_synthetic_sfm_pcd(ref_pcd, rotation_degrees=[5,5,5], translation=[0.1,0.2,0.05]) # , noise_level = 0.001)

# Align point clouds
ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)


# ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)
chamfer, hausdorff = compute_alignment_errors(ref_pcd, target_aligned)



# Generate synethic test point cloud (mimic sfm)
sfm_pcd = sfm_pipeline.generate_synthetic_sfm_pcd(ref_pcd, rotation_degrees=[5,5,5], translation=[0.1,0.2,0.05], noise_level = 0.01)

# Align point clouds
ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)


# ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)
chamfer, hausdorff = compute_alignment_errors(ref_pcd, target_aligned)



# TODO do voxel downsampling first?? 
# Or just set to 1 and skip downsampling since less than 100,000 points 













import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import itertools
import copy

def global_chamfer_align(ref_pcd, sfm_pcd, rotations=np.linspace(0, 360, 12), voxel_size=None):
    """
    Coarse alignment of sfm_pcd to ref_pcd by minimizing Chamfer distance over discrete rotations.
    Automatically centers the point clouds to handle large offsets.
    """

    ref_pts_orig = np.asarray(ref_pcd.points, dtype=np.float64)
    sfm_pts_orig = np.asarray(sfm_pcd.points, dtype=np.float64)

    # if voxel_size is not None:
    #     ref_pcd = ref_pcd.voxel_down_sample(voxel_size)
    #     sfm_pcd = sfm_pcd.voxel_down_sample(voxel_size)
    #     ref_pts_orig = np.asarray(ref_pcd.points, dtype=np.float64)
    #     sfm_pts_orig = np.asarray(sfm_pcd.points, dtype=np.float64)

    # --- Compute centroids and center clouds ---
    ref_centroid = np.mean(ref_pts_orig, axis=0)
    sfm_centroid = np.mean(sfm_pts_orig, axis=0)

    ref_pts = ref_pts_orig - ref_centroid
    sfm_pts = sfm_pts_orig - sfm_centroid

    best_cd = np.inf
    best_transform = np.eye(4)

    angles_deg = list(rotations)
    angle_combinations = itertools.product(angles_deg, repeat=3)

    for rx, ry, rz in angle_combinations:
        rad = np.deg2rad([rx, ry, rz])
        R = sfm_pcd.get_rotation_matrix_from_xyz(rad)

        # Apply rotation to centered SfM points
        sfm_pts_rot = (R @ sfm_pts.T).T  # still shape (N,3)

        # Ensure contiguous float64
        sfm_pts_rot = np.ascontiguousarray(sfm_pts_rot, dtype=np.float64)
        ref_pts_c = np.ascontiguousarray(ref_pts, dtype=np.float64)

        # Chamfer distance expects (N,3)
        cd = pcu.chamfer_distance(sfm_pts_rot, ref_pts_c)

        if cd < best_cd:
            best_cd = cd
            best_transform = np.eye(4)
            best_transform[:3, :3] = R
            # Translate back to original frame
            best_transform[:3, 3] = ref_centroid - (R @ sfm_centroid)

    print(f"[INFO] Best Chamfer distance: {best_cd:.6f}")
    print(f"[INFO] Best transform:\n{best_transform}")

    return best_transform




def align_with_chamfer_then_icp(ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100):
    """
    Align sfm_pcd to ref_pcd using global Chamfer-based coarse alignment + ICP refinement.
    """
    # --- Global search ---
    coarse_transform = global_chamfer_align(ref_pcd, sfm_pcd, voxel_size=voxel_size)

    sfm_aligned = copy.deepcopy(sfm_pcd).transform(coarse_transform)

    # --- ICP refinement ---
    if icp_threshold is None:
        # Use ~2% of object size
        ref_obb = ref_pcd.get_oriented_bounding_box()
        icp_threshold = np.linalg.norm(ref_obb.extent) * 0.02

    reg = o3d.pipelines.registration.registration_icp(
        sfm_aligned, ref_pcd, icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter)
    )

    final_transform = reg.transformation @ coarse_transform
    sfm_final = copy.deepcopy(sfm_pcd).transform(final_transform)

    return sfm_final, final_transform

# Generate synethic test point cloud (mimic sfm)
sfm_pcd = sfm_pipeline.generate_synthetic_sfm_pcd(ref_pcd, rotation_degrees=[5,2,4], translation=[4, 2, 3], noise_level = 0.001)

# sfm_final, final_transform = align_with_chamfer_then_icp(ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100)
sfm_final, final_transform = align_with_chamfer_then_icp(
    ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100
)

# --- Visualize after alignment ---
sfm_final.paint_uniform_color([0.8, 0.1, 0.1])
ref_vis = copy.deepcopy(ref_pcd)
ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
o3d.visualization.draw_geometries([ref_vis, sfm_final], window_name="After Alignment")


chamfer, hausdorff = compute_alignment_errors(ref_pcd, sfm_final)

