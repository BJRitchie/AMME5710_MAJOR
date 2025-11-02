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
store_path= "images/checkerbox_nasa2"
vid_path = 'images/checkerbox_nasa2.mp4'
sfm_save_path = "images/checkerbox_nasa2_sfm"

# Storage files 
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"
sat_model_path = "sat_model"
ref_path = "reference.ply"

# Make images from the video 
gen_images_from_vid( vid_path, store_path ) 

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

# Make SFM point cloud 
sfm_pipeline.make_reference_pcd( ref_path ) 

sfm_pipeline.resize_ims( store_path, 1200, 2 )
sfm_pipeline.prep_pointcloud() 
sfm_pipeline.make_pointcloud()
sfm_pipeline.clean_pointcloud( nb_points=50, radius=1) 
sfm_pipeline.plot_pointcloud()

# Save only the images that SfM used
sfm_pipeline.save_registered_images(output_folder=sfm_save_path)

# Save pipeline class 
sfm_pipeline.save(sfm_save_path+".pkl") 

# Checkerboard detection on images the SFM used 
cb = checkerboard.Checkerboard() 
cb.read_ims(sfm_save_path) 
cb.undistort_ims(grid_size=(3,3), cell_size=0.0096)
cb.plot_checkerboards() 
cb.save() 

# Load in pickle files
with open("sfm_pipeline.pkl", "rb") as f:
    sfm_pipeline = pickle.load(f)
    
with open("checkerboard.pkl", "rb") as f:
    cb = pickle.load(f)

print("Loaded saved SfM pipeline and checkerboard data")

# Checkerboard camera poses 
checker_rotations, checker_translations = cb.get_camera_poses()
checker_translations = np.array(checker_translations)[:, :, 0].T # turn into 3xN
checker_rotations = np.array(checker_rotations).transpose(1, 2, 0) # turn into 3xN

# Get SFM camera poses 
sfm_rotations, sfm_translations, pcd = sfm_pipeline.get_pointcloud_and_poses()
sfm_pcd = np.asarray(pcd.points).T 
sfm_translations = np.array(sfm_translations)[:, :, 0].T 
sfm_rotations = np.array(sfm_rotations).transpose(1, 2, 0)

# Load reference ply 
pcd = sfm_pipeline.get_reference_pcd( ref_path ) 
ref_pcd = np.asarray(pcd.points).T 
N = ref_pcd.shape[1]

# Apply transform 
sref = 0.0001
Rref = np.array([
    [ 1, 0, 0],
    [ 0, -1, 0],
    [ 0,  0, -1]])

tref = np.array([151.08, -25., -3.31]) 

transformed = np.empty_like(ref_pcd)
for i in range(N):
    transformed[:, i] = sref * ((Rref @ ref_pcd[:, i]) - tref)
ref_pcd = transformed

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

# Extract the matched poses 
matched_indices_sfm, matched_indices_cb = match_sfm_camera_poses(cb, sparse_path = "sparse/0")
sfm_translations_matched    = (sfm_translations[:, matched_indices_sfm])
sfm_rotations_matched       = (sfm_rotations[:, :, matched_indices_sfm])
checker_trans_matched       = (checker_translations[:, matched_indices_cb])
checker_rot_matched         = (checker_rotations[:, :, matched_indices_cb])

# Match pointclouds using camera poses 
from point_cloud_matcher import PointCloudMatcher 

pc_matcher = PointCloudMatcher() 
R, t, s, T, best_inliers = pc_matcher.matchFromPosesRANSAC( 
    poses0=sfm_translations_matched, 
    poses1=checker_trans_matched, 
    threshold=0.01, 
    ransac_samples=10 ) 

# R, t, s, T, best_inliers = pc_matcher.matchFromPosesOrientsRANSAC(
#     poses0=sfm_translations_matched,
#     rots0=sfm_rotations_matched, 
#     poses1=checker_trans_matched, 
#     rots1=checker_rot_matched,
#     w_rot=1,
#     ransac_samples=5, 
#     threshold=0.1, 
#     max_iter=1000
# )

# check SVD singular values:
print(f"Number of inliers: { len(best_inliers) }")

print("\n====== Umeyama ======") 
print("R: ") 
print(R)
print("\nt: ") 
print(t)
print("\ns: ") 
print(s)

# pc_matcher.transformPointClouds( np.full_like(sfm_pcd, np.nan), np.full_like(sfm_pcd, np.nan) )
# pc_matcher.transformPointClouds( sfm_pcd, ref_pcd )
pc_matcher.transformPointClouds( sfm_pcd, np.full_like(ref_pcd, np.nan) )
pc_matcher.transformCameraPoses( sfm_translations_matched, sfm_rotations_matched, checker_trans_matched, checker_rot_matched )
pc_matcher.plotMultiPointClouds( camera_scale=0.01 ) 


