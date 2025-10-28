import pycolmap
import matplotlib.pyplot as plt
import open3d as o3d

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

from plot_pointcloud_testing import plot_pointcloud_orig, plot_pointcloud_matched, plot_checkerboard_camera_poses, plot_sfm_vs_checkerboard, plot_pointcloud_scaled

# # Convert the video into images 
# vid_path = 'images/bigger_checker.mp4'
# store_path="images/bigger_checker"
# gen_images_from_vid( vid_path, store_path ) 

# # Storage files 
# im_path = store_path
# db_path = "database.db"
# sparse_path = "sparse"
# dense_path = "dense"

# # Settings 
# sift_ops = pycolmap.SiftExtractionOptions()
# sift_ops.use_gpu = False # CPU only 
# sift_ops.first_octave = 0
# sift_ops.num_octaves = 4

# # Initialise the pipeline 
# sfm_pipeline = pipeline.StrcFromMotion ( 
#     db_path, im_path, sparse_path, dense_path,
#     cam_mode    =pycolmap.CameraMode.AUTO, 
#     cam_model   ="SIMPLE_RADIAL",  
#     reader_ops  =pycolmap.ImageReaderOptions(), 
#     sift_ops    =sift_ops, 
#     device      =pycolmap.Device.cpu 
# ) 

# sfm_pipeline.resize_ims( store_path, 1200, 2 )  # originally 10 but wasn't using images with checkerboard in it
# sfm_pipeline.prep_pointcloud() 
# sfm_pipeline.make_pointcloud()
# sfm_pipeline.clean_pointcloud() # TODO experiment with this a bit more - gets rid of far outliers but not close ones (just off of body)
# sfm_pipeline.plot_pointcloud() 


# sfm_save_path = "images/bigger_checker_sfm"

# # Save only the images that SfM used
# sfm_pipeline.save_registered_images(output_folder=sfm_save_path)



# # Checkerboard detection on images the SFM used 
# cb = checkerboard.Checkerboard() 
# cb.read_ims(sfm_save_path) 
# cb.undistort_ims(grid_size=(3, 3), cell_size=0.0096)
# cb.plot_checkerboards() 

# plt.show()


# SAVE/LOAD
import pickle

# # ---------------------------------------------------------
# # 8. Save pipeline state for future reuse
# # ---------------------------------------------------------
# with open("sfm_pipeline.pkl", "wb") as f:
#     pickle.dump(sfm_pipeline, f)
# print("Saved SfM pipeline to 'sfm_pipeline.pkl'")

# with open("checkerboard.pkl", "wb") as f:
#     pickle.dump(cb, f)
# print("Saved checkerboard data to 'checkerboard.pkl'")

# LOAD 
with open("sfm_pipeline.pkl", "rb") as f:
    sfm_pipeline = pickle.load(f)

with open("checkerboard.pkl", "rb") as f:
    cb = pickle.load(f)

print("Loaded saved SfM pipeline and checkerboard data")




import open3d as o3d
import numpy as np
import cv2

def plot_in_checkerboard_frame(cb):
    """
    Visualize the camera poses in the checkerboard (satellite) frame.
    Each camera is shown relative to the fixed checkerboard.
    """
    assert hasattr(cb, "_rvecs") and hasattr(cb, "_tvecs"), "Run undistort_ims() first."

    # Create checkerboard 3D points (same as in calibration)
    objp = np.zeros((np.prod(cb._grid_size), 3), np.float32)
    objp[:, :2] = np.indices(cb._grid_size).T.reshape(-1, 2)
    objp *= cb._cell_size

    axis_length = float(cb._cell_size * cb._grid_size[0] / 2)
    geometries = []

    # --- Fixed checkerboard in world (board) frame ---
    pcd_board = o3d.geometry.PointCloud()
    pcd_board.points = o3d.utility.Vector3dVector(objp)
    pcd_board.paint_uniform_color([0.8, 0.8, 0.8])
    geometries.append(pcd_board)

    # Checkerboard frame
    board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 1.2)
    geometries.append(board_frame)

    # --- Camera poses in board frame ---
    for i, (rvec, tvec) in enumerate(zip(cb._rvecs, cb._tvecs)):
        R_cam_board, _ = cv2.Rodrigues(rvec)
        t_cam_board = np.asarray(tvec).reshape(3)

        # Invert to get camera pose in board frame
        R_board_cam = R_cam_board.T
        t_board_cam = -R_board_cam @ t_cam_board

        # Draw coordinate frame for this camera
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 0.8)
        cam_frame.rotate(R_board_cam, center=(0, 0, 0))
        cam_frame.translate(t_board_cam)
        geometries.append(cam_frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Cameras in Checkerboard (Satellite) Frame",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

plot_in_checkerboard_frame(cb)
















sparse_path = "sparse/0"
rec = pycolmap.Reconstruction(sparse_path)
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)

## GET SFM AND CHECKERBOARD CAMERA POSE ESTIMATES
# Get SFM pose estimates
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()

print(len(sfm_rotations))
print(len(sfm_translations))

# Extract checkerboard pose estimates 
# checker_rotations, checker_translations = cb.get_poses()
checker_rotations, checker_translations = cb.get_camera_poses()


print(len(checker_rotations))
print(len(checker_translations))


## EXTRACT MATCHING CHECKERBOARD POSES IN SFM 
import os
cb_images = cb._checker_image_names
matched_indices_sfm = []
matched_indices_cb = []


# REPAINT SATELLITE SOLAR PANELS
for j, cb_name in enumerate(cb_images):
    for i, sfm_img in enumerate(sfm_images):
        if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
            print(os.path.basename(sfm_img.name))
            matched_indices_sfm.append(i)
            matched_indices_cb.append(j)
            break

print(matched_indices_sfm)
print(matched_indices_cb) # DONT NEED



## ORIGINAL, UNSCALED POINT CLOUDS ## 
# All SFM poses
plot_pointcloud_orig(
    sparse_path="sparse",
    store_path="0",
    camera_scale=0.1,
)

# Only SFM poses that match checkerboard poses 
plot_pointcloud_matched("sparse", store_path="0", camera_scale=0.1, matched_indices=matched_indices_sfm)






import matplotlib.pyplot as plt
import numpy as np
import cv2

# -----------------------------
# 1. Gather checkerboard points and camera positions
# -----------------------------

# Get 3D points of all checkerboards
checker_points = []
for rvec, tvec in zip(cb._rvecs, cb._tvecs):
    R_board, _ = cv2.Rodrigues(rvec)
    t_board = np.asarray(tvec).reshape(3)
    objp = np.zeros((np.prod(cb._grid_size), 3), np.float32)
    objp[:, :2] = np.indices(cb._grid_size).T.reshape(-1, 2)
    objp *= cb._cell_size
    points_world = (R_board @ objp.T + t_board.reshape(3, 1)).T
    checker_points.append(points_world)

checker_points = np.vstack(checker_points)

# Get camera positions
camera_rotations, camera_positions = cb.get_camera_poses()
camera_positions = np.hstack(camera_positions).T  # shape (N, 3)

# -----------------------------
# 2. Plot point cloud + checkerboards + cameras
# -----------------------------

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# # Plot SfM scaled points if available
# if 'points_scaled' in globals():
#     ax.scatter(points_scaled[:,0], points_scaled[:,1], points_scaled[:,2],
#                c='gray', s=1, label='SfM points')

# Plot checkerboard points
ax.scatter(checker_points[:,0], checker_points[:,1], checker_points[:,2],
           c='blue', s=10, label='Checkerboard corners')

# Plot camera positions
ax.scatter(camera_positions[:,0], camera_positions[:,1], camera_positions[:,2],
           c='red', s=50, marker='^', label='Checkerboard cameras')

# Optional: draw camera axes
axis_length = cb._cell_size * max(cb._grid_size) * 0.5
for R_cam, C_cam in zip(camera_rotations, camera_positions):
    origin = C_cam
    x_axis = origin + axis_length * R_cam[:,0]
    y_axis = origin + axis_length * R_cam[:,1]
    z_axis = origin + axis_length * R_cam[:,2]
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], c='r')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], c='g')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], c='b')

# -----------------------------
# 3. Final plot settings
# -----------------------------

ax.set_box_aspect([1,1,1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.title("Checkerboards and Camera Poses in 3D")
ax.legend()
plt.show()





import numpy as np

# Matched SfM and checkerboard camera translations
sfm_translations_matched = [sfm_translations[i].reshape(3) for i in matched_indices_sfm]
checker_positions = np.hstack(cb.get_camera_poses()[1]).T  # shape (N,3) in meters

# Compute scale factor for each pair of cameras
scale_factors = []
n = len(sfm_translations_matched)

for i in range(n):
    for j in range(i + 1, n):  # avoid duplicate pairs and self-comparison
        # Distance in COLMAP units
        d_sfm = np.linalg.norm(sfm_translations_matched[j] - sfm_translations_matched[i])
        # Distance in meters from checkerboard
        d_m = np.linalg.norm(checker_positions[j] - checker_positions[i])
        scale_factors.append(d_m / d_sfm)

# Use the average scale factor
scale_factor = np.mean(scale_factors)
print(f"Robust scale factor from all camera pairs: {scale_factor:.4f} meters/unit")




# ---------------------------------------------------------
# Matplotlib plot to show scaled point cloud only
# ---------------------------------------------------------
print("Load and scale point cloud")
pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Apply scale factor
points_scaled = points * scale_factor

# Create a new point cloud object with scaled points
pcd_scaled = o3d.geometry.PointCloud()
pcd_scaled.points = o3d.utility.Vector3dVector(points_scaled)
pcd_scaled.colors = o3d.utility.Vector3dVector(colors)  # preserve original colors

# Save the scaled point cloud
o3d.io.write_point_cloud("sparse/0/points_scaled.ply", pcd_scaled)
print("Scaled point cloud saved to 'sparse/0/points_scaled.ply'")

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_scaled[:,0], points_scaled[:,1], points_scaled[:,2], c=colors, s=1)

ax.set_box_aspect([1,1,1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.title("Scaled Point Cloud (Metric Scale Applied)")
plt.show()

















# import sys
# sys.exit()














import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------------
# 1. Get matched camera positions
# ----------------------------------
# checkerboard camera positions in meters
_, cb_cams = cb.get_camera_poses()
cb_cams = np.hstack(cb_cams).T  # shape (N, 3)

# SfM camera positions
sfm_cams = np.array([sfm_translations[i].reshape(3) for i in matched_indices_sfm])  # (N,3)

# Apply scale factor to SfM cameras
sfm_cams_scaled = sfm_cams * scale_factor

# ----------------------------------
# 2. Compute rigid transform using Kabsch algorithm
# ----------------------------------
def compute_rigid_transform(A, B):
    """
    Computes rotation R and translation t to align A -> B (A and B are Nx3)
    Returns R (3x3) and t (3,)
    """
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[2, :] *= -1
        R_ = Vt.T @ U.T
    t_ = centroid_B - R_ @ centroid_A
    return R_, t_

R_align, t_align = compute_rigid_transform(cb_cams, sfm_cams_scaled)

# ----------------------------------
# 3. Transform checkerboard points to SfM frame
# ----------------------------------
checker_points = []
for rvec, tvec in zip(cb._rvecs, cb._tvecs):
    R_board, _ = cv2.Rodrigues(rvec)
    t_board = np.asarray(tvec).reshape(3)
    objp = np.zeros((np.prod(cb._grid_size), 3), np.float32)
    objp[:, :2] = np.indices(cb._grid_size).T.reshape(-1, 2)
    objp *= cb._cell_size
    points_world = (R_board @ objp.T + t_board.reshape(3, 1)).T
    # Transform to SfM frame
    points_sfm = (R_align @ points_world.T).T + t_align
    checker_points.append(points_sfm)

checker_points = np.vstack(checker_points)

# ----------------------------------
# 4. Transform checkerboard cameras to SfM frame
# ----------------------------------
cb_cams_sfm = (R_align @ cb_cams.T).T + t_align



def rotation_error(R1, R2):
    """Returns the angular difference (in degrees) between two rotation matrices."""
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    """Returns Euclidean distance between two translation vectors."""
    return np.linalg.norm(t1 - t2)


rotation_diffs = []
translation_diffs = []

# --- Get matched SfM and checkerboard poses (after alignment) ---
sfm_Rs = [sfm_rotations[i] for i in matched_indices_sfm]
sfm_ts = [sfm_translations[i].reshape(3) * scale_factor for i in matched_indices_sfm]

# Aligned checkerboard camera rotations and translations
cb_Rs, cb_ts = cb.get_camera_poses()
cb_Rs = [R_align @ R for R in cb_Rs]  # rotate into SfM frame
cb_ts = [(R_align @ t.reshape(3) + t_align) for t in cb_ts]

# --- Compute per-camera errors ---
for i in range(len(matched_indices_sfm)):
    R_sfm = sfm_Rs[i]
    t_sfm = sfm_ts[i]
    R_cb = cb_Rs[i]
    t_cb = cb_ts[i]

    r_err = rotation_error(R_sfm, R_cb)
    t_err = translation_error(t_sfm, t_cb)

    rotation_diffs.append(r_err)
    translation_diffs.append(t_err)

    filename = os.path.basename(cb._checker_image_names[matched_indices_cb[i]])
    print(f"{filename}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

# --- Summary statistics ---
print("\nPose validation summary:")
print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")
print(f"Max rotation error: {np.max(rotation_diffs):.3f}°")
print(f"Max translation error: {np.max(translation_diffs):.4f} m")









import open3d as o3d
import numpy as np
import cv2

def plot_checkerboard_and_sfm_in_cb_frame(cb, sfm_cams_scaled, R_align, t_align, scale_factor):
    """
    Visualize both checkerboard-based and SfM-estimated camera poses in the same
    checkerboard (satellite) coordinate frame using Open3D.
    
    Parameters
    ----------
    cb : CheckerboardCalibration
        Your checkerboard calibration object (must have _rvecs, _tvecs, _grid_size, _cell_size)
    sfm_cams_scaled : np.ndarray (N, 3)
        SfM camera centers scaled to real-world units
    R_align : np.ndarray (3, 3)
        Rotation aligning checkerboard to SfM
    t_align : np.ndarray (3,)
        Translation aligning checkerboard to SfM
    scale_factor : float
        Scale factor between SfM and real world (meters/unit)
    """

    assert hasattr(cb, "_rvecs") and hasattr(cb, "_tvecs"), "Run undistort_ims() first."
    assert hasattr(cb, "_grid_size") and hasattr(cb, "_cell_size"), "Checkerboard must be defined."

    # --------------------------------------------
    # Checkerboard geometry
    # --------------------------------------------
    objp = np.zeros((np.prod(cb._grid_size), 3), np.float32)
    objp[:, :2] = np.indices(cb._grid_size).T.reshape(-1, 2)
    objp *= cb._cell_size
    axis_length = float(cb._cell_size * cb._grid_size[0] / 2)

    geometries = []

    # Add the fixed checkerboard
    pcd_board = o3d.geometry.PointCloud()
    pcd_board.points = o3d.utility.Vector3dVector(objp)
    pcd_board.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(pcd_board)

    # Checkerboard frame (world origin)
    board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 1.5)
    geometries.append(board_frame)

    # --------------------------------------------
    # Plot camera poses from checkerboard calibration
    # --------------------------------------------
    for i, (rvec, tvec) in enumerate(zip(cb._rvecs, cb._tvecs)):
        R_cam_cb, _ = cv2.Rodrigues(rvec)
        t_cam_cb = np.asarray(tvec).reshape(3)

        # Convert camera to checkerboard frame
        R_cb_cam = R_cam_cb.T
        t_cb_cam = -R_cb_cam @ t_cam_cb

        # Draw camera coordinate frame (checkerboard-based)
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 0.8)
        cam_frame.rotate(R_cb_cam, center=(0, 0, 0))
        cam_frame.translate(t_cb_cam)
        geometries.append(cam_frame)

    # --------------------------------------------
    # Plot SfM cameras (transformed into checkerboard frame)
    # --------------------------------------------
    sfm_cams_in_cb = (R_align.T @ (sfm_cams_scaled.T - t_align.reshape(3, 1))).T

    for i, cam_pos in enumerate(sfm_cams_in_cb):
        # Small red sphere for each SfM camera
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.1)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.translate(cam_pos)
        geometries.append(sphere)

    # --------------------------------------------
    # Show everything
    # --------------------------------------------
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Checkerboard vs SfM Cameras (Checkerboard Frame)",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

    print(f"Plotted {len(cb._rvecs)} checkerboard cameras and {len(sfm_cams_scaled)} SfM cameras.")



plot_checkerboard_and_sfm_in_cb_frame(
    cb=cb,
    sfm_cams_scaled=sfm_cams_scaled,
    R_align=R_align,
    t_align=t_align,
    scale_factor=scale_factor
)








import sys
sys.exit()














## Trying to plot transformed checkerboards and cameras with scaled point cloud (but not working)
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# -----------------------------
# 1. Load scaled SfM point cloud
# -----------------------------

pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
points_sfm = np.asarray(pcd.points) * scale_factor
colors_sfm = np.asarray(pcd.colors)

# -----------------------------
# 2. Checkerboard to satellite transformation
# -----------------------------
R_cb_to_sat = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0, -1.0, -1.2246468e-16],
    [ 0.0,  1.2246468e-16, -1.0]
])
t_cb_to_sat = np.array([[151.08], [-25.0], [-3.31]]) / 1000.0  # meters

def cb_to_sat(points_cb):
    """Transform points from checkerboard frame to satellite frame."""
    return (R_cb_to_sat @ points_cb.T + t_cb_to_sat).T

# -----------------------------
# 3. Transform checkerboard points and cameras
# -----------------------------
# checker_points: (N,3) array of all checkerboard corner points in checkerboard frame
# camera_positions: (M,3) array of checkerboard camera positions in checkerboard frame
# camera_rotations: list of 3x3 camera rotations in checkerboard frame

checker_points_sat = cb_to_sat(checker_points)
cb_cams_sat = cb_to_sat(camera_positions)
camera_rotations_sat = [R_cb_to_sat @ R_cam for R_cam in camera_rotations]

# -----------------------------
# 4. Plot everything in Matplotlib
# -----------------------------
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

# Plot scaled SfM points
ax.scatter(points_sfm[:,0], points_sfm[:,1], points_sfm[:,2],
           c=colors_sfm, s=1, label='SfM points', alpha=0.5)

# Plot transformed checkerboard points
ax.scatter(checker_points_sat[:,0], checker_points_sat[:,1], checker_points_sat[:,2],
           c='blue', s=10, label='Checkerboard corners')

# Plot transformed checkerboard cameras
ax.scatter(cb_cams_sat[:,0], cb_cams_sat[:,1], cb_cams_sat[:,2],
           c='red', s=50, marker='^', label='Checkerboard cameras')

# Draw camera axes
axis_length = cb._cell_size * max(cb._grid_size) * 0.5
for R_cam, C_cam in zip(camera_rotations_sat, cb_cams_sat):
    origin = C_cam
    x_axis = origin + axis_length * R_cam[:,0]
    y_axis = origin + axis_length * R_cam[:,1]
    z_axis = origin + axis_length * R_cam[:,2]
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], c='r')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], c='g')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], c='b')

# Final plot settings
ax.set_box_aspect([1,1,1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Scaled SfM Point Cloud with Checkerboards and Cameras")
ax.legend()
plt.show()


















## EXTRACT MATCHED IMAGES
# Select only matched poses
sfm_rotations = [sfm_rotations[i] for i in matched_indices_sfm]
sfm_translations = [sfm_translations[i] for i in matched_indices_sfm]

print(f"Number of matched poses: {len(sfm_rotations)} rotations, {len(sfm_translations)} translations")




# # Not right 
# plot_pointcloud_checkerboard(
#     sparse_path="sparse/0",
#     checker_rotations=checker_rotations,
#     checker_translations=checker_translations,
#     matched_indices_cb=matched_indices_cb,
#     camera_scale=0.1
# )



import sys
sys.exit()
















import numpy as np

# ------------------------------
# 0. Get camera poses from checkerboard
# ------------------------------
camera_rotations, camera_positions = cb.get_camera_poses()  # from your Checkerboard class
# camera_positions: 3x1 vectors in checkerboard/world frame
# camera_rotations: rotation matrices from camera to checkerboard/world frame

# ------------------------------
# 1. Transform each camera pose into satellite frame
# ------------------------------

# R_cb_to_sat = np.array([
#     [ 6.12323400e-17, -1.00000000e+00, -1.22464680e-16],
#     [ -1.00000000e+000, -6.12323400e-17, -7.49879891e-33],
#     [0.00000000e+00, 1.22464680e-16, -1.00000000e+00]
# ])

# t_cb_to_sat = np.array([[1.4025e+02], [-1.4000e-01], [-3.5100e+00]]) / 1000.0  # mm → m


# other panel 
R_cb_to_sat = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0, -1.0, -1.2246468e-16],
    [ 0.0,  1.2246468e-16, -1.0]
])
t_cb_to_sat = np.array([[151.08], [-25.0], [-3.31]]) / 1000.0  # mm → m

# Transformation matrix from checkerboard to satellite frame
T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3, :3] = R_cb_to_sat
T_cb_to_sat[:3, 3] = t_cb_to_sat.flatten()

T_sat_to_cam_list = []
for R_cam, C_cam in zip(camera_rotations, camera_positions):
    # Build camera pose in checkerboard frame (4x4)
    T_cam_cb = np.eye(4)
    T_cam_cb[:3, :3] = R_cam
    T_cam_cb[:3, 3] = C_cam.flatten()

    # Transform into satellite frame
    T_sat_to_cam = T_cb_to_sat @ T_cam_cb  # satellite <- camera
    T_sat_to_cam_list.append(T_sat_to_cam)

# ------------------------------
# 2. Compare SfM poses with checkerboard camera poses
# ------------------------------
def rotation_error(R1, R2):
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

rotation_diffs = []
translation_diffs = []

# for i in range(len(sfm_rotations)):
#     R_sfm = sfm_rotations[i]
#     t_sfm = sfm_translations[i]

#     T_cb = T_sat_to_cam_list[i]
#     R_cb = T_cb[:3, :3]
#     t_cb = T_cb[:3, 3]

#     rotation_diffs.append(rotation_error(R_sfm, R_cb))
#     translation_diffs.append(translation_error(t_sfm, t_cb))

# matched_indices_cb contains indices of checkerboard images corresponding to SfM cameras
for i, idx_cb in zip(range(len(sfm_rotations)), matched_indices_cb):
    R_sfm = sfm_rotations[i]
    t_sfm = sfm_translations[i]

    T_cb = T_sat_to_cam_list[i]
    R_cb = T_cb[:3, :3]
    t_cb = T_cb[:3, 3]

    r_err = rotation_error(R_sfm, R_cb)
    t_err = translation_error(t_sfm, t_cb)

    filename = os.path.basename(cb._checker_image_names[idx_cb])
    print(f"{filename}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")


# ------------------------------
# 3. Report validation
# ------------------------------
# print("\nPose validation results:")
# for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
#     print(f"Camera {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")

# ------------------------------
# 4. Compute scale from matched translations
# ------------------------------
sfm_t_array = np.hstack([t for t in sfm_translations])
cb_t_array  = np.hstack([T[:3, 3].reshape(3, 1) for T in T_sat_to_cam_list])

sfm_t_centered = sfm_t_array - sfm_t_array[:, [0]]
cb_t_centered  = cb_t_array - cb_t_array[:, [0]]

scale = np.sum(cb_t_centered * sfm_t_centered) / np.sum(sfm_t_centered**2)
scale = abs(scale)
print(f"Estimated scale factor: {scale:.6f}")

sfm_translations_scaled = [scale * t for t in sfm_translations]

# Optional: recompute errors after scaling
translation_diffs_scaled = [
    translation_error(t_sfm_scaled, T_sat_to_cam_list[i][:3,3])
    for i, t_sfm_scaled in enumerate(sfm_translations_scaled[:len(T_sat_to_cam_list)])
]

print("\nTranslation errors after scaling:")
for i, t_err in enumerate(translation_diffs_scaled):
    print(f"Camera {i}: translation error = {t_err:.4f} m")

print(f"Mean translation error after scaling: {np.mean(translation_diffs_scaled):.4f} m")


plot_sfm_vs_checkerboard(sparse_path, 
                             sfm_rotations, 
                             sfm_translations_scaled, 
                             checker_rotations, 
                             checker_translations, 
                             T_cb_to_sat, 
                             matched_indices_cb=None, 
                             camera_scale=0.1)
























import numpy as np 

# ------------------------------
# 1. Transform checkerboard poses to satellite frame
# ------------------------------
# For old smaller checkerboard
# # Define checkerboard → satellite transform
# R_cb_to_sat = np.array([
#     [ 0.96917272, 0.0, 0.24638229],
#     [ 0.0,        1.0, 0.0      ],
#     [-0.24638229, 0.0, 0.96917272]
# ])

# t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]]) / 1000.0  # mm → m


# Define checkerboard → satellite transform
# R_cb_to_sat = np.array([
#     [ 6.12323400e-17, -1.00000000e+00, -1.22464680e-16],
#     [ -1.00000000e+000, -6.12323400e-17, -7.49879891e-33],
#     [0.00000000e+00, 1.22464680e-16, -1.00000000e+00]
# ])

# t_cb_to_sat = np.array([[1.4025e+02], [-1.4000e-01], [-3.5100e+00]]) / 1000.0  # mm → m

# Other panel
R_cb_to_sat = np.array([
    [ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
    [ 0.0000000e+00, -1.0000000e+00, -1.2246468e-16],
    [ 0.0000000e+00,  1.2246468e-16, -1.0000000e+00]
])
t_cb_to_sat = np.array([[151.08], [-25.0], [-3.31]]) / 1000.0  # mm → m



# Normal way: new = R * old + t 
# With transformation matrices: 
# T = [R, t; 0, 1]
# new = T * old 

T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3, :3] = R_cb_to_sat
T_cb_to_sat[:3, 3] = t_cb_to_sat.flatten()

# TODO CHECK THIS:
# Transform each checkerboard camera pose into satellite frame
T_sat_to_cam_list = []
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    T_cb_to_cam = np.eye(4)
    T_cb_to_cam[:3, :3] = R_cb
    T_cb_to_cam[:3, 3] = t_cb.flatten()
    
    T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
    T_sat_to_cam_list.append(T_sat_to_cam)


# ------------------------------
# 2. Compare SfM poses with checkerboard poses
# ------------------------------
def rotation_error(R1, R2):
    """Return angular difference (deg) between two rotation matrices."""
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    """Return Euclidean distance (m) between two translation vectors."""
    return np.linalg.norm(t1 - t2)

rotation_diffs = []
translation_diffs = []

for i in range(len(sfm_rotations)):
    R_sfm = sfm_rotations[i]
    t_sfm = sfm_translations[i]
    
    T_cb = T_sat_to_cam_list[i]
    R_cb = T_cb[:3, :3]
    t_cb = T_cb[:3, 3]

    rotation_diffs.append(rotation_error(R_sfm, R_cb))
    translation_diffs.append(translation_error(t_sfm, t_cb))

# ------------------------------
# 3. Report validation
# ------------------------------
print("\nPose validation results:")
for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Camera {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")






# ------------------------------
# 4. Compute scale from matched translations
# ------------------------------
# Stack translations into 3 x N arrays
sfm_t_array = np.hstack([t for t in sfm_translations])  # 3 x N (all SfM translations)
cb_t_array  = np.hstack([T[:3, 3].reshape(3, 1) for T in T_sat_to_cam_list])  # 3 x N (matched checkerboard poses)

# Center translations (remove origin offset)
sfm_t_centered = sfm_t_array - sfm_t_array[:, [0]]
cb_t_centered  = cb_t_array - cb_t_array[:, [0]]

# Compute scale factor
scale = np.sum(cb_t_centered * sfm_t_centered) / np.sum(sfm_t_centered**2)
print(f"Estimated scale factor: {scale:.6f}")

if scale < 0:
    scale *= -1

# Apply scale to all SfM translations
sfm_translations_scaled = [scale * t for t in sfm_translations]

# Optional: recompute errors after scaling
translation_diffs_scaled = [translation_error(t_sfm_scaled, T_sat_to_cam_list[i][:3,3])
                            for i, t_sfm_scaled in enumerate(sfm_translations_scaled[:len(T_sat_to_cam_list)])]

print("\nTranslation errors after scaling:")
for i, t_err in enumerate(translation_diffs_scaled):
    print(f"Camera {i}: translation error = {t_err:.4f} m")

print(f"Mean translation error after scaling: {np.mean(translation_diffs_scaled):.4f} m")









plot_checkerboard_camera_poses("sparse/0", T_sat_to_cam_list, camera_scale=0.05)

sfm_rotations_all, sfm_translations_all = sfm_pipeline.get_poses()
sfm_translations_all = [scale * t for t in sfm_translations_all]
plot_pointcloud_scaled("sparse", sfm_rotations_all, sfm_translations_all, store_path="0", camera_scale=0.1)

# No scaling to sfm translations
# plot_sfm_vs_checkerboard(sparse_path, 
#                              sfm_rotations, 
#                              sfm_translations, 
#                              checker_rotations, 
#                              checker_translations, 
#                              T_cb_to_sat, 
#                              matched_indices_cb=None, 
#                              camera_scale=0.1)


plot_sfm_vs_checkerboard(sparse_path, 
                             sfm_rotations, 
                             sfm_translations_scaled, 
                             checker_rotations, 
                             checker_translations, 
                             T_cb_to_sat, 
                             matched_indices_cb=None, 
                             camera_scale=0.1)




# ---------------------------------------------------------
# 7. Matplotlib plot to show scaled point cloud only
# ---------------------------------------------------------
print("Load and scale point cloud")
pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
points_scaled = points * scale  # apply scale


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_scaled[:,0], points_scaled[:,1], points_scaled[:,2], c=colors, s=1)

ax.set_box_aspect([1,1,1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.title("Scaled Point Cloud (Metric Scale Applied)")
plt.show()












print("\nNew method")





# FROM CHAT
import numpy as np
import os
import open3d as o3d
import pycolmap
import matplotlib.pyplot as plt

# ---------------------------
# 0. Helper functions
# ---------------------------
def compute_similarity_from_points(A, B, allow_scale=True):
    """
    Compute similarity (s, R, t) that maps points B -> A (A and B are 3xN).
    Uses Horn / Kabsch + isotropic scale.
    Returns s (scalar), R (3x3), t (3x1) such that A ≈ s * R @ B + t
    """
    assert A.shape[0] == 3 and B.shape[0] == 3 and A.shape[1] == B.shape[1]
    N = A.shape[1]
    mu_A = np.mean(A, axis=1, keepdims=True)
    mu_B = np.mean(B, axis=1, keepdims=True)

    A_c = A - mu_A
    B_c = B - mu_B

    # Cross-covariance
    H = B_c @ A_c.T   # note: we will compute R so that R @ B_c ≈ A_c => H = B_c A_c^T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # Fix reflection
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    if allow_scale:
        # scale s = trace(A_c^T R B_c) / sum(||B_c||^2)
        numerator = np.sum(A_c * (R @ B_c))
        denom = np.sum(B_c**2)
        s = float(numerator / denom) if denom > 0 else 1.0
    else:
        s = 1.0

    t = mu_A - s * R @ mu_B
    return s, R, t

def rotation_error_deg(R1, R2):
    dR = R1.T @ R2
    ang = np.arccos(np.clip((np.trace(dR)-1)/2, -1.0, 1.0))
    return np.degrees(ang)

def translation_error(t1, t2):
    return float(np.linalg.norm(t1 - t2))

def make_camera_frustum(R, t, width=640, height=480, fx=1, fy=1, scale=0.05, color=[1,0,0]):
    """Return Open3D LineSet for camera frustum using intrinsics (small pyramid)."""
    cx = width/2
    cy = height/2
    frustum_depth = scale * 2
    corners_cam = np.array([
        [0, 0, 0],
        [(0-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
        [(width-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
        [(width-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth],
        [(0-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth],
    ])
    corners_world = (R.T @ corners_cam.T).T + t.reshape(1,3)
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls

# ---------------------------
# 1. Build T_sat_to_cam_list (checkerboard cameras expressed in satellite frame)
#    (You already do this; here's the standard pattern.)
# ---------------------------
# Example: R_cb_to_sat and t_cb_to_sat must be provided (from CAD/SolidWorks)
# R_cb_to_sat = np.array([
#     [ 6.12323400e-17, -1.00000000e+00, -1.22464680e-16],
#     [ -1.00000000e+000, -6.12323400e-17, -7.49879891e-33],
#     [0.00000000e+00, 1.22464680e-16, -1.00000000e+00]
# ])

# t_cb_to_sat = np.array([[1.4025e+02], [-1.4000e-01], [-3.5100e+00]]) / 1000.0  # mm → m


# Other panel
R_cb_to_sat = np.array([
    [ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
    [ 0.0000000e+00, -1.0000000e+00, -1.2246468e-16],
    [ 0.0000000e+00,  1.2246468e-16, -1.0000000e+00]
])
t_cb_to_sat = np.array([[151.08], [-25.0], [-3.31]]) / 1000.0  # mm → m



T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3,:3] = R_cb_to_sat
T_cb_to_sat[:3,3]  = t_cb_to_sat.flatten()

# Build list of T_sat_to_cam (camera pose in satellite frame) from checker poses
T_sat_to_cam_list = []
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    # R_cb, t_cb are checkerboard->camera (as you've been using)
    T_cb_to_cam = np.eye(4)
    T_cb_to_cam[:3,:3] = R_cb
    T_cb_to_cam[:3,3]  = t_cb.flatten()
    # camera in satellite frame:
    T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
    T_sat_to_cam_list.append(T_sat_to_cam)

# ---------------------------
# 2. Extract matched SfM poses and matched checkerboard poses (centers)
# ---------------------------
# matched_indices_sfm and matched_indices_cb must already be computed by name matching
# Sanity check:
if len(matched_indices_sfm) != len(matched_indices_cb):
    raise RuntimeError("matched_indices lengths differ. Fix matching logic first.")

if len(matched_indices_sfm) < 3:
    raise RuntimeError("Need at least 3 matched poses for a robust similarity estimate.")

# SfM camera centers (3xN) from your original, full SfM list but only matched indices
sfm_rot_all, sfm_trans_all = sfm_pipeline.get_poses()
sfm_centers = np.hstack([sfm_trans_all[i].reshape(3,1) for i in matched_indices_sfm])  # 3 x N

# Checkerboard camera centers expressed in satellite frame (3xN), then pick matched ones
cb_centers_sat = np.hstack([T_sat_to_cam_list[j][:3,3].reshape(3,1) for j in matched_indices_cb])  # 3 x N

print("SfM centroid:", np.mean(sfm_centers, axis=1))
print("Checkerboard (sat) centroid:", np.mean(cb_centers_sat, axis=1))

# ---------------------------
# 3. Compute similarity transform that maps checkerboard centers -> SfM centers
# ---------------------------
s_est, R_align, t_est = compute_similarity = compute_similarity_from_points(sfm_centers, cb_centers_sat, allow_scale=True)
s = s_est; R = R_align; t = t_est  # s (scalar), R (3x3), t (3x1)
print(f"Estimated similarity: scale={s:.6e}, det(R)={np.linalg.det(R):.6f}")
print("t (m):", t.flatten())

# Diagnostics: errors before/after mapping
def compute_center_errors(sfm_centers, cb_centers_sat, s, R, t):
    N = sfm_centers.shape[1]
    errs_before = [np.linalg.norm(sfm_centers[:,i] - cb_centers_sat[:,i]) for i in range(N)]
    mapped = s * (R @ cb_centers_sat) + t
    errs_after  = [np.linalg.norm(sfm_centers[:,i] - mapped[:,i]) for i in range(N)]
    return np.array(errs_before), np.array(errs_after), mapped

errs_before, errs_after, cb_centers_mapped = compute_center_errors(sfm_centers, cb_centers_sat, s, R, t)
print("Mean center error before:", errs_before.mean())
print("Mean center error after :", errs_after.mean())

# ---------------------------
# 4. Apply mapping to full checkerboard poses (so they are in SfM frame)
#    For each T_sat_to_cam (4x4), map rotation and translation appropriately:
#    t_sfm = s * R @ t_sat + t_est
#    R_sfm = R @ R_sat   (no scale on rotation)
# ---------------------------
T_sfm_to_cam_mapped = []
for T_sat_to_cam in T_sat_to_cam_list:
    R_sat_cam = T_sat_to_cam[:3,:3]
    t_sat_cam = T_sat_to_cam[:3,3].reshape(3,1)
    R_mapped = R @ R_sat_cam
    t_mapped = (s * (R @ t_sat_cam)).reshape(3) + t.reshape(3)
    T_m = np.eye(4)
    T_m[:3,:3] = R_mapped
    T_m[:3,3]  = t_mapped
    T_sfm_to_cam_mapped.append(T_m)

# ---------------------------
# 5. Optional: compute rotation + translation errors vs SfM for matched poses
# ---------------------------
rotation_diffs = []
translation_diffs = []
for i_local, (i_sfm, j_cb) in enumerate(zip(matched_indices_sfm, matched_indices_cb)):
    R_sfm = sfm_rot_all[i_sfm]
    t_sfm = sfm_trans_all[i_sfm].reshape(3)
    T_mapped = T_sfm_to_cam_mapped[j_cb]  # mapped checkerboard pose
    R_m = T_mapped[:3,:3]
    t_m = T_mapped[:3,3]

    rotation_diffs.append(rotation_error_deg(R_sfm, R_m))
    translation_diffs.append(translation_error(t_sfm, t_m))

    print(f"R_sfm:{R_sfm}")
    print(f"t_sfm:{t_sfm}")
    print(f"R_cb:{R_m}")
    print(f"t_cb:{t_m}\n")



print("\nMatched-pose validation after mapping (checkerboard -> SfM):")
for i,(r_err,t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Image {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")
print("Mean rotation error:", np.mean(rotation_diffs))
print("Mean translation error:", np.mean(translation_diffs))

# ---------------------------
# 6. Visualize: scaled point cloud (SfM) + SfM cameras (red) + mapped checkerboard cameras (green)
# ---------------------------
# Load and scale SfM point cloud (if you estimated a scale to apply to the whole SfM cloud)
ply_path = os.path.join("sparse","0","points.ply")
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if len(pcd.colors)>0 else None

# If you want to apply scale to point cloud so distances match checkerboard (optional)
# We'll use s to scale points if you intend to render everything in SfM frame consistent with the mapping
points_scaled = points * s
pcd_scaled = o3d.geometry.PointCloud()
pcd_scaled.points = o3d.utility.Vector3dVector(points_scaled)
if colors is not None:
    pcd_scaled.colors = o3d.utility.Vector3dVector(colors)

geoms = [pcd_scaled]

# Add SfM cameras (use original SfM centers and rotations)
rec = pycolmap.Reconstruction(os.path.join("sparse","0"))
sfm_images_sorted = sorted(rec.images.values(), key=lambda x: x.name)
sfm_cameras = {cam_id: cam for cam_id, cam in rec.cameras.items()}

# We'll only draw the matched SfM cameras to keep visual uncluttered
for idx_local, i_sfm in enumerate(matched_indices_sfm):
    img = sfm_images_sorted[i_sfm]
    cam_from_world = img.cam_from_world()
    R_sfm = cam_from_world.rotation.matrix()
    t_sfm = img.projection_center().flatten() * s  # scaled for visualization
    cam = sfm_cameras[img.camera_id]
    fx = cam.params[0] if len(cam.params) > 0 else 1.0
    fy = cam.params[1] if len(cam.params) > 1 else fx
    w,h = cam.width, cam.height

    # red for SfM
    geoms.append(make_camera_frustum(R_sfm, t_sfm, width=w, height=h, fx=fx, fy=fy, scale=0.1, color=[1,0,0]))

# Add mapped checkerboard cameras (green)
for j_cb, Tm in enumerate(T_sfm_to_cam_mapped):
    Rm = Tm[:3,:3]
    tm = Tm[:3,3]
    # use same intrinsics as an associated SfM camera if available (safe fallback to defaults)
    # Try to get intrinsics from matched SfM camera for visual consistency if indices align
    # Here j_cb corresponds to the checker index in full list; find matching matched index if exists
    # For visual, we use dummy intrinsics or neighbor SfM intrinsics.
    geoms.append(make_camera_frustum(Rm, tm, width=640, height=480, fx=500, fy=500, scale=0.1, color=[0,1,0]))

print("Launching Open3D visualizer: SfM (red) vs mapped checkerboard (green)")
o3d.visualization.draw_geometries(geoms, window_name="SfM vs Mapped Checkerboard")

# ---------------------------
# End
# ---------------------------
































import cv2

def plot_checkerboards_and_sfm(
    grid_size,
    cell_size,
    rvecs,
    tvecs,
    sfm_rotations,
    sfm_translations,
    checker_rotations=None,
    checker_translations=None,
    match_ref_index=0
):
    """
    Visualize the checkerboard calibration poses (from rvecs/tvecs)
    and SfM poses (rotations/translations) together in 3D space.

    Args:
        grid_size: tuple (cols, rows) of checkerboard corners.
        cell_size: float, cell edge length (same units as tvecs).
        rvecs: list of rotation vectors from cv2 calibration.
        tvecs: list of translation vectors from cv2 calibration.
        sfm_rotations: list of 3x3 numpy arrays from SfM.
        sfm_translations: list of 3x1 numpy arrays from SfM.
        checker_rotations: list of 3x3 rotation matrices for checkerboard (optional).
        checker_translations: list of 3x1 translations for checkerboard (optional).
        match_ref_index: index of which checkerboard to use as reference
                         for transforming SfM poses into checkerboard frame.
    """
    # === Build checkerboard corner coordinates ===
    objp = np.zeros((np.prod(grid_size), 3), np.float32)
    objp[:, :2] = np.indices(grid_size).T.reshape(-1, 2)
    objp *= cell_size

    geometries = []

    # === Axis scaling ===
    axis_length = float(cell_size * grid_size[0] / 2)

    # Camera origin frame
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 1.2)
    geometries.append(cam_frame)

    colors = [
        [0.8, 0.2, 0.2],
        [0.2, 0.8, 0.2],
        [0.2, 0.2, 0.8],
        [0.8, 0.6, 0.2],
        [0.6, 0.2, 0.8],
        [0.2, 0.8, 0.6]
    ]

    # === Plot checkerboard calibration poses ===
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rvec)
        # origin = np.asarray(tvec).reshape(3).astype(float)
        # board_points = (R @ objp.T + origin.reshape(3, 1)).T.astype(float)

        # Ensure tvec is a column vector shape (3,1)
        R = R.reshape(3, 3)
        tvec = tvec.reshape(3, 1)
        objp = objp.reshape(-1, 3)
        print(R.shape, objp.shape, tvec.shape)
        
        board_points = (R @ objp.T + tvec).T


        # tvec = np.asarray(tvec, dtype=float).reshape(3, 1)
        # board_points = (R @ objp.T + tvec).T
        origin = tvec.flatten()


        # Checkerboard points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(board_points)
        pcd.paint_uniform_color(colors[i % len(colors)])
        geometries.append(pcd)

        # Checkerboard frame
        board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 0.6)
        board_frame.rotate(R, center=(0, 0, 0))
        board_frame.translate(origin)
        geometries.append(board_frame)

    # === Plot SfM poses transformed into checkerboard frame ===
    if checker_rotations is not None and checker_translations is not None:
        R_cb_ref = checker_rotations[match_ref_index]
        t_cb_ref = checker_translations[match_ref_index]
        T_cb_ref = np.eye(4)
        T_cb_ref[:3, :3] = R_cb_ref
        T_cb_ref[:3, 3] = t_cb_ref.flatten()

        for R_sfm, t_sfm in zip(sfm_rotations, sfm_translations):
            T_sfm = np.eye(4)
            T_sfm[:3, :3] = R_sfm
            T_sfm[:3, 3] = t_sfm.flatten()

            # Transform SfM camera pose into checkerboard frame
            T_cam_in_cb = np.linalg.inv(T_cb_ref) @ np.linalg.inv(T_sfm)
            R_cam_in_cb = T_cam_in_cb[:3, :3]
            t_cam_in_cb = T_cam_in_cb[:3, 3]

            # Plot SfM camera as a smaller coordinate frame
            cam_frame_sfm = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 0.4)
            cam_frame_sfm.rotate(R_cam_in_cb, center=(0, 0, 0))
            cam_frame_sfm.translate(t_cam_in_cb)
            geometries.append(cam_frame_sfm)

    # === Visualize all geometries ===
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Checkerboard + SfM Poses (Checkerboard Frame)",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )



plot_checkerboards_and_sfm(
    grid_size=(3, 3),
    cell_size=0.0096,
    rvecs=checker_rotations,
    tvecs=checker_translations,
    sfm_rotations=sfm_rotations,
    sfm_translations=sfm_translations,
    checker_rotations=checker_rotations,
    checker_translations=checker_translations,
    match_ref_index=0  # pick whichever checkerboard you want as reference
)
