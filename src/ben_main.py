# use generate_test_pcds_sat 

import pycolmap
import pickle
import open3d as o3d
import numpy as np
import cv2 
import os 

import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

# Convert the video into images 
# CHANGE TO WHATEVER VIDEO/POINT CLOUD YOU HAVE 
store_path= "images/ps4_controller"
vid_path = 'images/ps4_controller.mp4'
sfm_save_path = "images/ps4_controller_sfm"

# Storage files 
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"
sat_model_path = "sat_model"

# # Make images from the video 
# gen_images_from_vid( vid_path, store_path ) 

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

# # Make SFM point cloud 
# # sfm_pipeline.make_reference_ply()
# # sfm_pipeline.plot_reference_model()

# sfm_pipeline.resize_ims( store_path, 1200, 3 )
# sfm_pipeline.prep_pointcloud() 
# sfm_pipeline.make_pointcloud()
# sfm_pipeline.clean_pointcloud(nb_points=75, radius=5) 
# sfm_pipeline.plot_pointcloud()


# # Save only the images that SfM used
# sfm_pipeline.save_registered_images(output_folder=sfm_save_path)


# # Checkerboard detection on images the SFM used 
# cb = checkerboard.Checkerboard() 
# cb.read_ims(sfm_save_path) 
# cb.undistort_ims(grid_size=(3,3), cell_size=0.116)
# cb.plot_checkerboards() 

# # Save pipeline state for future reuse
# with open("sfm_pipeline.pkl", "wb") as f:
#     pickle.dump(sfm_pipeline, f)
# print("Saved SfM pipeline to 'sfm_pipeline.pkl'")

# with open("checkerboard.pkl", "wb") as f:
#     pickle.dump(cb, f)
# print("Saved checkerboard data to 'checkerboard.pkl'")

# Load in pickle files
with open("sfm_pipeline.pkl", "rb") as f:
    sfm_pipeline = pickle.load(f)

with open("checkerboard.pkl", "rb") as f:
    cb = pickle.load(f)

print("Loaded saved SfM pipeline and checkerboard data")


# Load an example pointcloud 
from gen_synthetic_pcd_lib import generate_test_pcds_sat 
ref_pcd = generate_test_pcds_sat()
ref_pcd = np.asarray(ref_pcd.points).T

# Checkerboard camera poses 
checker_rotations, checker_translations = cb.get_camera_poses()
checker_translations = np.array(checker_translations)[:, :, 0].T # turn into 3xN
checker_rotations = np.array(checker_rotations).transpose(1, 2, 0) # turn into 3xN

# Construct paths
store_name = os.path.join(sparse_path, '0')
file_path = os.path.join(store_name, "points.ply")

# Load point cloud
if not os.path.exists(file_path):
    raise FileNotFoundError(f"points.ply not found at {file_path}")

# SFM points 
pcd = o3d.io.read_point_cloud(file_path) # Output of SFM 
sfm_pcd = np.asarray(pcd.points).T 

# Get SFM camera poses 
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()
sfm_translations = np.array(sfm_translations)[:, :, 0].T 
sfm_rotations = np.array(sfm_rotations).transpose(1, 2, 0)

def match_sfm_camera_poses(cb, sparse_path="sparse/0"):
    import os
    import pycolmap

    rec = pycolmap.Reconstruction(sparse_path)
    images_sorted = sorted(rec.images.values(), key=lambda x: x.name)
    name_to_index = {os.path.basename(img.name): i for i, img in enumerate(images_sorted)}
    index_to_name = {i: os.path.basename(img.name) for i, img in enumerate(images_sorted)}

    matched_indices_sfm = []
    matched_indices_cb = []

    print("===== Matched Image Pairs =====")
    print(f"{'Checkerboard Image':40s}  |  {'SfM Image':40s}")

    for j, cb_name in enumerate(cb._checker_image_names):
        cb_base = os.path.basename(cb_name)
        
        if cb_base in name_to_index:
            sfm_idx = name_to_index[cb_base]
            sfm_name = index_to_name[sfm_idx]
            print(f"{cb_base:40s}  |  {sfm_name:40s}")

            matched_indices_sfm.append(sfm_idx)
            matched_indices_cb.append(j)

    print("\nTotal matched pairs:", len(matched_indices_sfm))
    return np.array(matched_indices_sfm, dtype=int), np.array(matched_indices_cb, dtype=int)

matched_indices_sfm = match_sfm_camera_poses(cb, sparse_path = "sparse/0")
# sfm_translations_matched = np.array([sfm_translations[:, idx] for idx in matched_indices_sfm]).T
# sfm_rotations_matched = np.array([sfm_rotations[:, :, idx] for idx in matched_indices_sfm]).transpose(1, 2, 0)

matched_indices_sfm, matched_indices_cb = match_sfm_camera_poses(cb, sparse_path = "sparse/0")

sfm_translations_matched    = (sfm_translations[:, matched_indices_sfm])[:, :20]
sfm_rotations_matched       = (sfm_rotations[:, :, matched_indices_sfm])[:, :, :20]
checker_trans_matched       = (checker_translations[:, matched_indices_cb])[:, :20]
checker_rot_matched         = (checker_rotations[:, :, matched_indices_cb])[:, :, :20]

# Match pointclouds using camera poses 
from point_cloud_matcher import PointCloudMatcher 

pc_matcher = PointCloudMatcher() 
R, t, s, T, best_inliers = pc_matcher.matchFromPosesRANSAC( 
    poses0=sfm_translations_matched, 
    poses1=checker_trans_matched, 
    threshold=0.1, 
    ransac_samples=6 )

# check SVD singular values:
print(f"Number of inliers: { len(best_inliers) }")

print("\n====== Umeyama ======") 
print("R: ") 
print(R)
print("\nt: ") 
print(t)
print("\ns: ") 
print(s)

pc_matcher._s = s
# pc_matcher._t = np.zeros((1, 3)) #t
# pc_matcher._R = rotation_mat_degs(roll=0, pitch=45, yaw=45) 

# T = np.eye(4)
# T[:3, :3] = np.eye(3) #s * R 
# T[:3, 3] = np.zeros((3,)) #t.flatten() 
# pc_matcher._T = T 

# pc_matcher.transformPointClouds( np.full_like(sfm_pcd, np.nan), np.full_like(sfm_pcd, np.nan) )
pc_matcher.transformPointClouds( sfm_pcd, np.full_like(ref_pcd, np.nan) )
pc_matcher.transformCameraPoses( sfm_translations_matched, sfm_rotations_matched, checker_trans_matched, checker_rot_matched )
pc_matcher.plotPointClouds( camera_scale=0.01 ) 


