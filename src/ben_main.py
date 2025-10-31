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
store_path= "images/checker_nasa_box"
vid_path = 'images/checker_nasa_box.mp4'
sfm_save_path = "images/checker_nasa_box_sfm"
# gen_images_from_vid( vid_path, store_path ) 

# Storage files 
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"
sat_model_path = "sat_model"

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
# sfm_pipeline.make_reference_ply()
# sfm_pipeline.plot_reference_model()

# sfm_pipeline.resize_ims( store_path, 1200, 1 )
# sfm_pipeline.prep_pointcloud() 
# sfm_pipeline.make_pointcloud()
# sfm_pipeline.clean_pointcloud() 
# sfm_pipeline.plot_pointcloud()


# # Save only the images that SfM used
# sfm_pipeline.save_registered_images(output_folder=sfm_save_path)


# # Checkerboard detection on images the SFM used 
# cb = checkerboard.Checkerboard() 
# cb.read_ims(sfm_save_path) 
# cb.undistort_ims(grid_size=(3, 3), cell_size=0.0096)
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




# Checkerboard camera poses 
# checker_rotations, checker_translations = cb.get_camera_poses()
# checker_translations = np.array(checker_translations)[:, :, 0].T # turn into 3xN
# checker_rotations = np.array(checker_rotations).transpose(1, 2, 0) # turn into 3xN

# Old function to plot so doesn't use get_camera_poses but still visualises it 
# def plot_in_checkerboard_frame(cb):
#     """
#     Visualize the camera poses in the checkerboard (satellite) frame.
#     Each camera is shown relative to the fixed checkerboard.
#     """
#     assert hasattr(cb, "_rvecs") and hasattr(cb, "_tvecs"), "Run undistort_ims() first."

#     # Create checkerboard 3D points (same as in calibration)
#     objp = np.zeros((np.prod(cb._grid_size), 3), np.float32)
#     objp[:, :2] = np.indices(cb._grid_size).T.reshape(-1, 2)
#     objp *= cb._cell_size

#     axis_length = float(cb._cell_size * cb._grid_size[0] / 2)
#     geometries = []

#     # --- Fixed checkerboard in world (board) frame ---
#     pcd_board = o3d.geometry.PointCloud()
#     pcd_board.points = o3d.utility.Vector3dVector(objp)
#     pcd_board.paint_uniform_color([0.8, 0.8, 0.8])
#     geometries.append(pcd_board)

#     # Checkerboard frame
#     board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 1.2)
#     geometries.append(board_frame)

#     # --- Camera poses in board frame ---
#     for i, (rvec, tvec) in enumerate(zip(cb._rvecs, cb._tvecs)):
#         R_cam_board, _ = cv2.Rodrigues(rvec)
#         t_cam_board = np.asarray(tvec).reshape(3)

#         # Invert to get camera pose in board frame
#         R_board_cam = R_cam_board.T
#         t_board_cam = -R_board_cam @ t_cam_board

#         # Draw coordinate frame for this camera
#         cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 0.8)
#         cam_frame.rotate(R_board_cam, center=(0, 0, 0))
#         cam_frame.translate(t_board_cam)
#         geometries.append(cam_frame)

#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="Cameras in Checkerboard (Satellite) Frame",
#         width=1024,
#         height=768,
#         mesh_show_back_face=True
#     )

# plot_in_checkerboard_frame(cb)

# Construct paths
store_name = os.path.join(sparse_path, '0')
file_path = os.path.join(store_name, "points.ply")

# Load point cloud
if not os.path.exists(file_path):
    raise FileNotFoundError(f"points.ply not found at {file_path}")
pcd = o3d.io.read_point_cloud(file_path) # Output of SFM 

# SFM points 
sfm_pcd = np.asarray(pcd.points).T 

# Get SFM camera poses 
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()
sfm_translations = np.array(sfm_translations)[:, :, 0].T 
sfm_rotations = np.array(sfm_rotations).transpose(1, 2, 0)

# sfm_translations_matched = sfm_translations_matched[:, :, 0].T
# sfm_rotations_matched = sfm_rotations_matched.transpose(1, 2, 0)

# Apply a synthetic translation and rotation 
# Rotation about X axis
roll = np.deg2rad(45)
Rx = np.array([ [1, 0, 0], 
                [0, np.cos(roll), -np.sin(roll)], 
                [0, np.sin(roll), np.cos(roll)] ])

# Rotation about Y axis
pitch = np.deg2rad(72)    
Ry = np.array([ [np.cos(pitch), 0, np.sin(pitch)], 
                [0, 1, 0], 
                [-np.sin(pitch), 0, np.cos(pitch)] ])

# Rotation about Z axis
yaw = np.deg2rad(132)      
Rz = np.array([ [np.cos(yaw), -np.sin(yaw), 0], 
                [np.sin(yaw), np.cos(yaw), 0], 
                [0, 0, 1 ] ]) 

s_true = 2
R_true = Rz @ Ry @ Rx 
t_true = np.empty((3,1))
t_true[:,0] = [5, 10, 15]

c0 = sfm_translations
r0 = sfm_rotations


N = c0.shape[1]
synth_translations = np.empty_like(c0)
synth_rotations = np.empty_like(sfm_rotations)
synth_pcd = np.empty_like(sfm_pcd) 

for i in range(N):
    
    # Camera poses 
    cam_pnt = c0[:, i]
    cam_rot = r0[:, :, i]
    synth_translations[:, i] = s_true * (R_true @ cam_pnt) + t_true.T
    synth_rotations[:, :, i] = R_true @ cam_rot
    
for i in range( sfm_pcd.shape[1] ):
    # Pointcloud 
    sfm_pnt = sfm_pcd[:, i] 
    synth_pcd[:, i] = s_true * (R_true @ sfm_pnt) + t_true.T 

print("====== TRUE ======") 
print("R_true: ") 
print(R_true)
print("\nt_true: ") 
print(t_true)
print("\ns_true: ") 
print(s_true)


# Restrict to SFM camera poses to just checkerboard cameras 
# def match_sfm_camera_poses(cb, sparse_path = "sparse/0"):
#     rec = pycolmap.Reconstruction(sparse_path)
#     # sfm_images = sorted(rec.images.values(), key=lambda x: x.name) # sort by how good they are, but don't need here 
#     sfm_images = rec.images.values()

#     cb_images = cb._checker_image_names
#     matched_indices_sfm = []
#     matched_indices_cb = []

#     for j, cb_name in enumerate(cb_images):
#         for i, sfm_img in enumerate(sfm_images):
#             if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
#                 print(os.path.basename(sfm_img.name))
#                 matched_indices_sfm.append(i)
#                 matched_indices_cb.append(j)
#                 break

#     return matched_indices_sfm

#     # print(matched_indices_sfm)
#     # print(matched_indices_cb) 

# matched_indices_sfm = match_sfm_camera_poses(cb, sparse_path = "sparse/0")
# sfm_translations_matched = np.array([sfm_translations[:][idx] for idx in matched_indices_sfm])
# sfm_translations_matched = sfm_translations_matched[:, :, 0].T # turn into 3xN

# sfm_rotations_matched = np.array([sfm_rotations[:][:][idx] for idx in matched_indices_sfm])
# sfm_rotations_matched = sfm_rotations_matched.transpose(1, 2, 0) # turn into 3x3xN

# Plot unscaled SFM with just the cmaeras matched to checkerboard cameras 
# def plot_pointcloud_matched(sparse_path, store_path="0", camera_scale=0.1, matched_indices=None):
#     """
#     Visualize a COLMAP sparse reconstruction (points + camera frustums), optionally only for matched poses.

#     Args:
#         sparse_path (str): Path to the parent sparse directory (e.g., "path/to/sparse").
#         store_path (str): Subfolder name (e.g., "0") containing the reconstruction and points.ply.
#         camera_scale (float): Scale factor for visualizing camera coordinate frames and frustums.
#         matched_indices (list[int], optional): List of indices of images to plot. If None, plots all cameras.
#     """
#     print("=== Loading and visualizing sparse point cloud with cameras ===")

#     # Construct paths
#     store_name = os.path.join(sparse_path, store_path)
#     file_path = os.path.join(store_name, "points.ply")

#     # Load point cloud
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"points.ply not found at {file_path}")
#     pcd = o3d.io.read_point_cloud(file_path)

#     # Load reconstruction (cameras + poses)
#     rec = pycolmap.Reconstruction(store_name)

#     # Sort images to make indexing consistent
#     images_sorted = sorted(rec.images.values(), key=lambda x: x.name)

#     # If matched_indices is not provided, plot all images
#     if matched_indices is None:
#         matched_indices = list(range(len(images_sorted)))

#     # Prepare visualization geometries
#     geometries = [pcd]

#     for idx in matched_indices:
#         image = images_sorted[idx]
#         cam_from_world = image.cam_from_world()
#         R = cam_from_world.rotation.matrix()  # world → camera
#         t = image.projection_center().flatten()  # camera center in world space

#         # --- Coordinate frame ---
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         T = np.eye(4)
#         T[:3, :3] = R.T  # world orientation
#         T[:3, 3] = t
#         camera_frame.transform(T)
#         geometries.append(camera_frame)

#         # --- Camera frustum ---
#         camera = rec.cameras[image.camera_id]
#         width, height = camera.width, camera.height
#         frustum_depth = camera_scale * 2

#         params = camera.params
#         if len(params) >= 2:
#             fx, fy = params[0], params[1]
#         else:
#             fx = fy = params[0] if len(params) > 0 else width

#         cx = width / 2
#         cy = height / 2

#         corners_cam = np.array([
#             [0, 0, 0],  # camera origin
#             [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#             [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#         ])

#         corners_world = (R.T @ corners_cam.T).T + t

#         lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
#         colors = [[1, 0, 0] for _ in lines]

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)

#         geometries.append(line_set)

#     print(f"Visualizing {len(matched_indices)} cameras and {len(pcd.points)} points")
#     o3d.visualization.draw_geometries(geometries)

# plot_pointcloud_matched("sparse", store_path="0", camera_scale=0.1, matched_indices=matched_indices_sfm)



# Match pointclouds using camera poses 
from point_cloud_matcher import PointCloudMatcher 

pc_matcher = PointCloudMatcher() 
R, t, s, T = pc_matcher.matchFromPoses( sfm_translations, synth_translations )

print("\n====== Umeyama ======") 
print("R: ") 
print(R)
print("\nt: ") 
print(t)
print("\ns: ") 
print(s)

# pc_matcher._s = 1.0
# pc_matcher._t = t
# pc_matcher._R = np.eye(3)#R 
# pc_matcher._T = T 

pc_matcher.transformPointClouds( sfm_pcd, synth_pcd )
pc_matcher.transformCameraPoses( sfm_translations, sfm_rotations, synth_translations, synth_rotations )
pc_matcher.plotPointClouds()


# # ------ BUNDLE ADJUSTMENT METHOD ------ # 

# from scipy.optimize import least_squares
# from scipy.spatial.transform import Rotation as Rsc

# def residual(params, poses0, poses1):
#     # params = [rx, ry, rz, tx, ty, tz, s]
#     rvec = params[:3]
#     t = params[3:6]
#     s = params[6]
#     R = Rsc.from_rotvec(rvec).as_matrix()
#     err = s * (R @ poses0) + t[:, None] - poses1
#     return err.flatten()

# def nonlinear_alignment(poses0, poses1):
#     params0 = np.zeros(7)  # initial guess: no rotation, no translation, scale=1
#     params0[6] = 1.0
#     res = least_squares(residual, params0, args=(poses0, poses1))
#     rvec, t, s = res.x[:3], res.x[3:6], res.x[6]
#     R = Rsc.from_rotvec(rvec).as_matrix()
#     return R, t, s

# R, t, s = nonlinear_alignment( sfm_translations, synth_translations ) 

# T = np.eye(4)
# T[:3, :3] = s * R
# T[:3, 3] = t.flatten()

# print("\n====== Bundle Adjustment ======") 
# print("R: ") 
# print(R)
# print("\nt: ") 
# print(t)
# print("\ns: ") 
# print(s)

# pc_matcher2 = PointCloudMatcher() 

# pc_matcher2._s = 1.0
# pc_matcher2._t = t
# pc_matcher2._R = np.eye(3)#R 
# pc_matcher2._T = T 
# pc_matcher2._poses0 = synth_translations 
# pc_matcher2._poses1 = sfm_translations 

# pc_matcher2.transformPointClouds( pcd )
# pc_matcher2.transformCameraPoses( sfm_translations, sfm_rotations, synth_translations, synth_rotations)
# pc_matcher2.plotPointClouds()


