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


# sfm_pipeline.make_reference_ply()
# sfm_pipeline.plot_reference_model()

# sfm_pipeline.resize_ims( store_path, 1200, 10 )
# sfm_pipeline.prep_pointcloud() 
# sfm_pipeline.make_pointcloud()
# sfm_pipeline.clean_pointcloud() 
# sfm_pipeline.plot_pointcloud()



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



# print("No noise\n")

# # Generate synethic test point cloud (mimic sfm)
# sfm_pcd = sfm_pipeline.generate_synthetic_sfm_pcd(ref_pcd, rotation_degrees=[5,5,5], translation=[0.1,0.2,0.05]) # , noise_level = 0.001)

# # Align point clouds
# ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)


# # ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)
# chamfer, hausdorff = compute_alignment_errors(ref_pcd, target_aligned)


# print("Bit of noise\n")
# # Generate synethic test point cloud (mimic sfm)
# sfm_pcd = sfm_pipeline.generate_synthetic_sfm_pcd(ref_pcd, rotation_degrees=[5,5,5], translation=[0.1,0.2,0.05], noise_level = 0.01)

# # Align point clouds
# ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)


# # ref_pcd, target_aligned = sfm_pipeline.align_pcds(ref_pcd, sfm_pcd)
# chamfer, hausdorff = compute_alignment_errors(ref_pcd, target_aligned)



# # TODO do voxel downsampling first?? 
# # Or just set to 1 and skip downsampling since less than 100,000 points 













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

    print("After coarse alginment")
    chamfer, hausdorff = compute_alignment_errors(ref_pcd, sfm_aligned)
    # sfm_aligned.paint_uniform_color([0.8, 0.1, 0.1])
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
    o3d.visualization.draw_geometries([ref_vis, sfm_aligned], window_name="After Coarse Alignment")


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

# # Generate synethic test point cloud (mimic sfm)
# sfm_pcd = sfm_pipeline.generate_synthetic_sfm_pcd(ref_pcd, rotation_degrees=[5,2,4], translation=[4, 2, 3], noise_level = 0.001)

# # sfm_final, final_transform = align_with_chamfer_then_icp(ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100)
# sfm_final, final_transform = align_with_chamfer_then_icp(
#     ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100
# )

# # --- Visualize after alignment ---
# sfm_final.paint_uniform_color([0.8, 0.1, 0.1])
# ref_vis = copy.deepcopy(ref_pcd)
# ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
# o3d.visualization.draw_geometries([ref_vis, sfm_final], window_name="After Alignment")


# chamfer, hausdorff = compute_alignment_errors(ref_pcd, sfm_final)






















ref_pcd = gen.generate_test_pcds_sat()

sfm_pcd = o3d.io.read_point_cloud(r"sparse\0\points_cleaned.ply")
print(f"Number of points: {len(sfm_pcd.points)}")

# # SCALE BY BOUNDING BOXES
# # --- Compute OBBs ---
# ref_obb = ref_pcd.get_oriented_bounding_box()
# sfm_obb = sfm_pcd.get_oriented_bounding_box()

# print(f"\nReference OBB dimensions: {ref_obb.extent}")
# print(f"Original SfM OBB dimensions: {sfm_obb.extent}")

# # --- Scale SfM cloud to match reference size ---
# scale_factor = np.linalg.norm(ref_obb.extent) / np.linalg.norm(sfm_obb.extent)
# sfm_center = sfm_obb.center
# sfm_pcd.translate(-sfm_center)
# sfm_pcd.scale(scale_factor, center=(0, 0, 0))

# # Recompute OBB after scaling & translation
# sfm_obb = sfm_pcd.get_oriented_bounding_box()
# print(f"Scaled SfM OBB dimensions: {sfm_obb.extent}")

# # --- Plot 1: point clouds only ---
# ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
# o3d.visualization.draw_geometries(
#     [ref_vis, sfm_pcd],
#     window_name="Reference (green) vs SfM (original colors)",
#     width=900, height=700
# )

# # --- Plot 2: point clouds with OBBs ---
# ref_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(ref_obb)
# ref_obb_ls.paint_uniform_color([0, 1, 0])  # green wireframe

# sfm_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(sfm_obb)
# sfm_obb_ls.paint_uniform_color([1, 0, 0])  # red wireframe

# o3d.visualization.draw_geometries(
#     [ref_vis, ref_obb_ls, sfm_pcd, sfm_obb_ls],
#     window_name="Reference (green) vs SfM (original colors) + OBBs",
#     width=900, height=700
# )


# SCALE FROM BEN 
# scale_factor = 0.06906822580219837
# centroid = sfm_pcd.get_center()
# sfm_pcd.scale(scale_factor, center=centroid)
# ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
# o3d.visualization.draw_geometries(
#     [ref_vis, sfm_pcd],
#     window_name="Reference (green) vs SfM (original colors)",
#     width=900, height=700
# )


# SCALE FROM OLLIE 
# Load COLMAP reconstruction
import os
sparse_path = "sparse/0"
rec = pycolmap.Reconstruction(sparse_path)
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)

# ---------------------------------------------------------
# Get SFM and Checkerboard poses
# ---------------------------------------------------------
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()
checker_rotations, checker_translations = cb.get_camera_poses()

# ---------------------------------------------------------
# Match corresponding image names
# ---------------------------------------------------------
cb_images = cb._checker_image_names
matched_indices_sfm = []
matched_indices_cb = []

for j, cb_name in enumerate(cb_images):
    for i, sfm_img in enumerate(sfm_images):
        if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
            matched_indices_sfm.append(i)
            matched_indices_cb.append(j)
            break

# ---------------------------------------------------------
# Compute robust scale factor between matched poses
# ---------------------------------------------------------
sfm_translations_matched = [sfm_translations[i].reshape(3) for i in matched_indices_sfm]
checker_positions = np.hstack(cb.get_camera_poses()[1]).T  # shape (N,3) in meters

scale_factors = []
n = len(sfm_translations_matched)

for i in range(n):
    for j in range(i + 1, n):
        d_sfm = np.linalg.norm(sfm_translations_matched[j] - sfm_translations_matched[i])
        d_m = np.linalg.norm(checker_positions[j] - checker_positions[i])
        if d_sfm > 1e-9:  # avoid division by zero
            scale_factors.append(d_m / d_sfm)

scale_factor = np.mean(scale_factors)
print(f"\nEstimated metric scale factor: {scale_factor:.6f} meters per COLMAP unit")

centroid = sfm_pcd.get_center()
sfm_pcd.scale(scale_factor, center=centroid)
ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
o3d.visualization.draw_geometries(
    [ref_vis, sfm_pcd],
    window_name="Reference (green) vs SfM (original colors)",
    width=900, height=700
)


sfm_final, final_transform = align_with_chamfer_then_icp(
    ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100
)

# --- Visualize after alignment ---
# sfm_final.paint_uniform_color([0.8, 0.1, 0.1])
ref_vis = copy.deepcopy(ref_pcd)
ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
o3d.visualization.draw_geometries([ref_vis, sfm_final], window_name="After Alignment")


chamfer, hausdorff = compute_alignment_errors(ref_pcd, sfm_final)

