import pycolmap
import matplotlib.pyplot as plt

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

# # Convert the video into images 
# vid_path = 'images/small_checker_spin.mp4'
# store_path="images/small_checker_spin"
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


# sfm_save_path = "images/small_checker_spin_sfm"

# # Save only the images that SfM used
# sfm_pipeline.save_registered_images(output_folder=sfm_save_path)



# # Checkerboard detection on images the SFM used 
# cb = checkerboard.Checkerboard() 
# cb.read_ims(sfm_save_path) 
# cb.undistort_ims(grid_size=(3, 3), cell_size=0.002)
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






# Get SFM pose estimates
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()

print(len(sfm_rotations))
print(len(sfm_translations))

# Extract checkerboard pose estimates 
checker_rotations, checker_translations = cb.get_poses()





import numpy as np
import open3d as o3d
import os

# Define transformation from checkerboard to satellite frame
# R_cb_to_sat = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.0, 0.0])
# t_cb_to_sat = np.array([[0.05], [0.02], [0.01]])

R_cb_to_sat = np.array([
    [ 0.96917272, 0.0, 0.24638229],
    [ 0.0,        1.0, 0.0      ],
    [-0.24638229, 0.0, 0.96917272]
])
t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]])
t_cb_to_sat = t_cb_to_sat / 1000.0  # if SolidWorks outputs mm


# Build homogeneous transform
T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3, :3] = R_cb_to_sat
T_cb_to_sat[:3, 3:] = t_cb_to_sat

# Convert checkerboard poses to satellite frame
T_sat_to_cam_list = []
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    T_cb_to_cam = np.eye(4)
    T_cb_to_cam[:3, :3] = R_cb
    T_cb_to_cam[:3, 3:] = t_cb
    # satellite → cam = cb → cam × sat → cb⁻¹
    T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
    T_sat_to_cam_list.append(T_sat_to_cam)

# ---------------------------------------------------------
# Align SfM and Checkerboard poses by image names
# ---------------------------------------------------------
rec_path = os.path.join(sfm_pipeline._sparse_path, "0")
rec = pycolmap.Reconstruction(rec_path)
sfm_images = list(rec.images.values())  # SfM images

cb_images = cb._checker_image_names  # checkerboard-detected image filenames

matched_indices_sfm = []
matched_indices_cb = []

for j, cb_name in enumerate(cb_images):
    for i, sfm_img in enumerate(sfm_images):
        if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
            matched_indices_sfm.append(i)
            matched_indices_cb.append(j)
            break

# Filter both datasets to only matched images
sfm_rotations = [sfm_rotations[i] for i in matched_indices_sfm]
sfm_translations = [sfm_translations[i] for i in matched_indices_sfm]
T_sat_to_cam_list = [T_sat_to_cam_list[j] for j in matched_indices_cb]

print(f"Matched {len(sfm_rotations)} SfM poses with checkerboard detections.")

# Debugging
print("SfM camera centers (m):")
for t in sfm_translations:
    print(t.flatten())

print("Checkerboard camera centers (m):")
for T in T_sat_to_cam_list:
    print(T[:3,3].flatten())


# ---------------------------------------------------------
# Compute and Apply Scale
# ---------------------------------------------------------
sfm_t_array = np.hstack([t for t in sfm_translations])               # 3 x N
cb_t_array  = np.hstack([T[:3, 3:] for T in T_sat_to_cam_list])      # 3 x N

s = np.sum(cb_t_array * sfm_t_array) / np.sum(sfm_t_array**2)
print(f"Estimated scale factor: {s:.4f}")

sfm_translations_scaled = [s * t for t in sfm_translations]


# ---------------------------------------------------------
# 4. Compute differences between SfM and checkerboard-derived poses
# ---------------------------------------------------------
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
num_images = len(T_sat_to_cam_list)  # matches SfM-used images

for i in range(num_images):
    R_sfm, t_sfm = sfm_rotations[i], sfm_translations_scaled[i]
    T_sat_cam_est = T_sat_to_cam_list[i]

    R_sat_cam = T_sat_cam_est[:3, :3]
    t_sat_cam = T_sat_cam_est[:3, 3:]

    rot_err = rotation_error(R_sfm, R_sat_cam)
    trans_err = translation_error(t_sfm, t_sat_cam)

    rotation_diffs.append(rot_err)
    translation_diffs.append(trans_err)

# ---------------------------------------------------------
# 5. Report validation results
# ---------------------------------------------------------
print("Pose validation results:")
for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Image {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"\nMean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")



# ---------------------------------------------------------
# Load scaled SfM point cloud
# ---------------------------------------------------------
print("Load and scale point cloud")
pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
points_scaled = points * s  # apply scale

scaled_pcd = o3d.geometry.PointCloud()
scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
scaled_pcd.colors = o3d.utility.Vector3dVector(colors)

geometries = [scaled_pcd]

# Frustum parameters
camera_scale = 0.02
frustum_depth = camera_scale * 2

def make_camera_frustum(R, t, width=640, height=480, fx=1, fy=1, scale=frustum_depth, color=[1,0,0]):
    """Create an Open3D LineSet representing the camera frustum."""
    cx = width / 2
    cy = height / 2
    corners_cam = np.array([
        [0, 0, 0],  # camera center
        [(0 - cx) * scale / fx, (0 - cy) * scale / fy, scale],
        [(width - cx) * scale / fx, (0 - cy) * scale / fy, scale],
        [(width - cx) * scale / fx, (height - cy) * scale / fy, scale],
        [(0 - cx) * scale / fx, (height - cy) * scale / fy, scale],
    ])
    corners_world = (R.T @ corners_cam.T).T + t
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    colors_lines = [color for _ in lines]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors_lines)
    return ls

# ---------------------------------------------------------
# Load SfM reconstruction
# ---------------------------------------------------------
rec = pycolmap.Reconstruction("sparse/0")
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)
sfm_cameras = {cam_id: cam for cam_id, cam in rec.cameras.items()}


for i in range(len(sfm_translations_scaled)):
    img = sfm_images[matched_indices_sfm[i]]  # get the corresponding SfM image
    t = sfm_translations_scaled[i].flatten()  # already matched & scaled
    T_cb = T_sat_to_cam_list[i]

    R = img.cam_from_world().rotation.matrix()

    cam = sfm_cameras[img.camera_id]
    fx = cam.params[0] if len(cam.params) > 0 else 1
    fy = cam.params[1] if len(cam.params) > 1 else fx
    width, height = cam.width, cam.height

    ls_sfm = make_camera_frustum(R, t, width, height, fx, fy, color=[1,0,0])
    geometries.append(ls_sfm)

    # Checkerboard camera
    T_cb = T_sat_to_cam_list[i]
    R_cb = T_cb[:3,:3]
    t_cb = T_cb[:3,3]
    ls_cb = make_camera_frustum(R_cb, t_cb, color=[0,1,0])
    geometries.append(ls_cb)

# ---------------------------------------------------------
# Visualize everything
# ---------------------------------------------------------
o3d.visualization.draw_geometries(
    geometries,
    window_name="Scaled Point Cloud + Camera Poses (SfM=Red, Checkerboard=Green)"
)



# ---------------------------------------------------------
# Add SfM camera frustums (scaled translations)
# ---------------------------------------------------------
# for img in sfm_images:
#     R = img.cam_from_world().rotation.matrix()
#     t = img.projection_center().flatten() * s  # apply scale
#     cam = sfm_cameras[img.camera_id]
#     fx = cam.params[0] if len(cam.params) > 0 else 1
#     fy = cam.params[1] if len(cam.params) > 1 else fx
#     width, height = cam.width, cam.height

#     ls = make_camera_frustum(R, t, width, height, fx, fy, color=[1,0,0])
#     geometries.append(ls)

# # ---------------------------------------------------------
# # Add checkerboard camera frustums
# # ---------------------------------------------------------
# for T in T_sat_to_cam_list:
#     R_cb = T[:3,:3]
#     t_cb = T[:3,3]
#     ls = make_camera_frustum(R_cb, t_cb, color=[0,1,0])
#     geometries.append(ls)

# # ---------------------------------------------------------
# # Visualize everything
# # ---------------------------------------------------------
# o3d.visualization.draw_geometries(
#     geometries,
#     window_name="Scaled Point Cloud + Camera Poses (SfM=Red, Checkerboard=Green)"
# )




# # ---------------------------------------------------------
# # 6. Visualize scaled point cloud + camera poses together (Open3D)
# # ---------------------------------------------------------
# def make_frame(T, color):
#     """Create Open3D coordinate frame geometry from a 4x4 pose matrix."""
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
#     frame.paint_uniform_color(color)
#     frame.transform(T)
#     return frame

# # Apply scale to the SfM point cloud
# print("Load point cloud")
# pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
# points = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors)
# points_scaled = points * s  # scale the points

# scaled_pcd = o3d.geometry.PointCloud()
# scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
# scaled_pcd.colors = o3d.utility.Vector3dVector(colors)

# # Build camera frames (SfM scaled, checkerboard)
# sfm_frames = [
#     make_frame(np.vstack([np.hstack([R, s * t]), [0, 0, 0, 1]]), [1, 0, 0])
#     for R, t in zip(sfm_rotations, sfm_translations)
# ]
# cb_frames = [make_frame(T, [0, 1, 0]) for T in T_sat_to_cam_list]

# # Visualize all together
# o3d.visualization.draw_geometries(
#     [scaled_pcd] + sfm_frames + cb_frames,
#     window_name="Scaled Point Cloud + Camera Poses (SfM=Red, Checkerboard=Green)"
# )


# ---------------------------------------------------------
# 7. Matplotlib plot to show scaled point cloud only
# ---------------------------------------------------------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_scaled[:,0], points_scaled[:,1], points_scaled[:,2], c=colors, s=1)

ax.set_box_aspect([1,1,1])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.title("Scaled Point Cloud (Metric Scale Applied)")
plt.show()

