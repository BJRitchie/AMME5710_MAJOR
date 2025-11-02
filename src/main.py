import pycolmap
 
import sfm_pipeline_lib as pipeline 
import gen_synthetic_pcd_lib as gen
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

import numpy as np
import open3d as o3d

import point_cloud_utils as pcu 
import itertools
import copy

import sys
import pickle

# TODO Integrate functions into class
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

def global_chamfer_align(ref_pcd, sfm_pcd, rotations=np.linspace(0, 360, 12), voxel_size=None):
    """
    Coarse alignment of sfm_pcd to ref_pcd by minimizing Chamfer distance over discrete rotations.
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

# TODO potentially experiment with parameters more
def align_with_chamfer_then_icp(ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100):
    """
    Align sfm_pcd to ref_pcd using global Chamfer-based coarse alignment + ICP refinement.
    """
    # --- Global search ---
    print("Starting coarse alignment...\n")
    coarse_transform = global_chamfer_align(ref_pcd, sfm_pcd, voxel_size=voxel_size)

    sfm_aligned = copy.deepcopy(sfm_pcd).transform(coarse_transform)

    print("After coarse alignment")
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

    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
    o3d.visualization.draw_geometries([ref_vis, sfm_final], window_name="After Alignment")

    return sfm_final, final_transform

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



# BEN TODO 
# Integrate scaling factor code 
# Check checkerboard initialisation is right/all necessary for your new code
# Add in anything else you want from your camera pose visualisation if we want an example of it 

# "SAVE" mode untested (wanted to each lunch and didn't want to wait lol)

# FLOW (Can use this for diagram) 
# (Simple overview, may want more steps in some spots - I'm also highlighting mostly the new alignment parts 
# I wrote since you already know SFM, Checkerboard class steps
# Read in images --> SFM --> Checkerboard --> (your process to scale point cloud) --> Scale Point Cloud
# --> Align Reference and SFM Point Clouds --> Performance Metrics 

# Align Reference and SFM Point Clouds: 
# Centre point clouds --> For different rotations, choose best coarse alignment to reference (minimises chamfer distance)
# --> ICP to refine alignment (ICP needs initial guess/global adjustment, can run from start since can get stuck in local minimum)


# TODO (later today - do before using new models)
# Fix reference generation - atm is generating a box point cloud, haven't changed back to using make_refernece
# - I think will just need to change what the reference file is then call that same function and it'll work
# - But also it is stored in the class so need to change how things are called/returned to use in later functions
# - that aren't integrated into the class yes 

# TODO (not necessary today for code to run)
# Move new functions into class and delete all old alignment ones
# Turn visualisations etc into functions in class 




# Change between SAVE and LOAD to generate/load SFM and checkerboard pickle files
mode = "LOAD" 
img_interval = 1 # Include every nth image from video into SFM generation


# File paths - TODO replace naming with concatenations 
store_path= "images/checker_nasa_box"
vid_path = 'images/checker_nasa_box.mp4'
sfm_save_path = "images/checker_nasa_box_sfm"
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"
sat_model_path = "sat_model"

# Convert the video into images 
if mode == "SAVE":
    gen_images_from_vid( vid_path, store_path ) 

    # Settings 
    sift_ops = pycolmap.SiftExtractionOptions()
    sift_ops.use_gpu = False # CPU only 
    sift_ops.first_octave = 0
    sift_ops.num_octaves = 4

    # Generate SFM class to initialise the pipeline
    sfm_pipeline = pipeline.StrcFromMotion ( 
        db_path, im_path, sparse_path, dense_path, sat_model_path,
        cam_mode    =pycolmap.CameraMode.AUTO, 
        cam_model   ="SIMPLE_RADIAL",  
        reader_ops  =pycolmap.ImageReaderOptions(), 
        sift_ops    =sift_ops, 
        device      =pycolmap.Device.cpu 
    ) 

    sfm_pipeline.make_reference_pcd() # TODO Change this/change solidwork file it points to 
    sfm_pipeline.plot_reference_model()
    
    sfm_pipeline.generate_and_plot_pointcloud(
        store_path, 1200, img_interval, 
        nb_pts1= 30, nb_pts2=60 
    )
    sfm_pipeline.save() 

    # TODO Ben: check this against how you've used/called the checkerboard, change however you need

    # Save only the images that SfM used (for checkerboard) 
    sfm_pipeline.save_registered_images(output_folder=sfm_save_path)

    # Checkerboard detection on images the SFM used 
    cb = checkerboard.Checkerboard() 
    cb.read_ims(sfm_save_path) 
    cb.undistort_ims(grid_size=(3, 3), cell_size=0.0096)
    cb.plot_checkerboards() 
    cb.save() 


elif mode == "LOAD":
    with open("sfm_pipeline.pkl", "rb") as f:
        sfm_pipeline = pickle.load(f)
    with open("checkerboard.pkl", "rb") as f:
        cb = pickle.load(f)
    print("Loaded saved SfM pipeline and checkerboard data")

else:
    print("Incorrect mode")
    sys.exit()

# Generate reference point cloud - TODO replace with sfm_pipeline.make_reference() that makes a reference point cloud from SolidWorks file
ref_pcd = gen.generate_test_pcds_sat()

# Load in point cloud cleaned from outliers - TODO Replace with proper integration/class function
sfm_pcd = o3d.io.read_point_cloud(r"sparse\0\points.ply") # Actual use this in final 
print(f"SFM Model point cloud loaded\n")

############# Camera Pose Matching #############

from point_cloud_matcher import PointCloudMatcher 
pc_matcher = PointCloudMatcher( sfm_pipeline, cb ) 
R, t, s, T = pc_matcher.matchPoints( sparse_path+"/0" )
pc_matcher.plotMultiPointClouds( camera_scale=0.01 ) 

############# Pointcloud Matching #############

# Scale SFM point cloud
scale_factor = s # TODO Replace with integrated function 
centroid = sfm_pcd.get_center()
sfm_pcd.scale(scale_factor, center=centroid)

# Visualisation of reference and scaled SFM - TODO make into function + centre them?
ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
o3d.visualization.draw_geometries(
    [ref_vis, sfm_pcd],
    window_name="Reference and Scaled SfM Model",
    width=900, height=700
)

# Align reference and SFM point clouds 
sfm_final, final_transform = align_with_chamfer_then_icp(
    ref_pcd, sfm_pcd, voxel_size=None, icp_threshold=None, icp_max_iter=100
)

# Compute final error metrics
print("\nFinal metrics")
chamfer, hausdorff = compute_alignment_errors(ref_pcd, sfm_final)

