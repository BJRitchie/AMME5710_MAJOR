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










# def chamfer_distance(pcd1, pcd2):
#     """
#     Compute symmetric Chamfer distance between two point clouds.
#     Args:
#         pcd1, pcd2: open3d.geometry.PointCloud
#     Returns:
#         float: Chamfer distance (mean symmetric)
#     """
#     # One direction: pcd1 → pcd2
#     d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
#     # Reverse direction: pcd2 → pcd1
#     d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
#     chamfer = np.mean(d1**2) + np.mean(d2**2)
#     return chamfer


# def hausdorff_distance(pcd1, pcd2):
#     """
#     Compute symmetric Hausdorff distance between two point clouds.
#     Args:
#         pcd1, pcd2: open3d.geometry.PointCloud
#     Returns:
#         float: Hausdorff distance (symmetric max distance)
#     """
#     # One direction: each point in pcd1 to nearest in pcd2
#     d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
#     # Reverse direction
#     d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
#     hausdorff = max(np.max(d1), np.max(d2))
#     return hausdorff


# cd = chamfer_distance(ref_pcd, target_aligned)
# hd = hausdorff_distance(ref_pcd, target_aligned)

# print(f"Chamfer distance:   {cd:.6f}")
# print(f"Hausdorff distance: {hd:.6f}")

