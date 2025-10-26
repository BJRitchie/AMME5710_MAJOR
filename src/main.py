import pycolmap
import matplotlib.pyplot as plt

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

from plot_pointcloud_testing import plot_pointcloud_orig, plot_pointcloud_matched, plot_checkerboard_camera_poses, plot_sfm_vs_checkerboard

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


sparse_path = "sparse/0"
rec = pycolmap.Reconstruction(sparse_path)
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)

## GET SFM AND CHECKERBOARD CAMERA POSE ESTIMATES
# Get SFM pose estimates
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()

print(len(sfm_rotations))
print(len(sfm_translations))

# Extract checkerboard pose estimates 
checker_rotations, checker_translations = cb.get_poses()

print(len(checker_rotations))
print(len(checker_translations))


## EXTRACT MATCHING CHECKERBOARD POSES IN SFM 
import os
cb_images = cb._checker_image_names
matched_indices_sfm = []
matched_indices_cb = []

for j, cb_name in enumerate(cb_images):
    for i, sfm_img in enumerate(sfm_images):
        if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
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






import numpy as np 

# ------------------------------
# 1. Transform checkerboard poses to satellite frame
# ------------------------------
# Define checkerboard → satellite transform
R_cb_to_sat = np.array([
    [ 0.96917272, 0.0, 0.24638229],
    [ 0.0,        1.0, 0.0      ],
    [-0.24638229, 0.0, 0.96917272]
])
t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]]) / 1000.0  # mm → m

T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3, :3] = R_cb_to_sat
T_cb_to_sat[:3, 3] = t_cb_to_sat.flatten()

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


import sys
sys.exit()
























import numpy as np

### COMPUTE SCALE
num_cameras = len(matched_indices_sfm)

# ------------------------------
# Compute scale between SfM and checkerboard translations
# ------------------------------
sfm_t_array = np.hstack(sfm_translations)
cb_t_array = np.hstack(checker_translations)

sfm_centered = sfm_t_array - sfm_t_array[:, [0]]
cb_centered = cb_t_array - cb_t_array[:, [0]]

s = np.sum(cb_centered * sfm_centered) / np.sum(sfm_centered**2)
sfm_translations_scaled = [s * t for t in sfm_translations]

print(f"Estimated scale factor: {s:.6f}")

if s < 0:
    print("Negative scale detected — flipping coordinate system.")
    s = -s
print(f"Estimated scale factor: {s:.6f}")

# # ------------------------------
# # Compute rotation alignment (optional)
# # ------------------------------
# def compute_rotation_alignment(S, C):
#     H = S @ C.T
#     U, _, Vt = np.linalg.svd(H)
#     R_align = Vt.T @ U.T
#     if np.linalg.det(R_align) < 0:
#         Vt[2,:] *= -1
#         R_align = Vt.T @ U.T
#     return R_align

# R_align = compute_rotation_alignment(np.hstack(sfm_translations_scaled), cb_t_array)
# sfm_rotations_aligned = [R_align @ R for R in sfm_rotations]
# sfm_translations_aligned = [R_align @ t for t in sfm_translations_scaled]

# ------------------------------
# Compute rotation and translation errors
# ------------------------------
def rotation_error(R1, R2):
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

rotation_diffs = []
translation_diffs = []
for i in range(num_cameras):
    # rotation_diffs.append(rotation_error(sfm_rotations_aligned[i], checker_rotations[i]))
    # translation_diffs.append(translation_error(sfm_translations_aligned[i], checker_translations[i]))

    rotation_diffs.append(rotation_error(sfm_rotations[i], checker_rotations[i]))
    translation_diffs.append(translation_error(sfm_translations_scaled[i], checker_translations[i]))


print("\nPose validation results:")
for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Camera {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")


print("\n")

































import numpy as np
import pycolmap
import open3d as o3d
import os
import pickle

# ------------------------------
# Load SfM pipeline and checkerboard data
# ------------------------------
with open("sfm_pipeline.pkl", "rb") as f:
    sfm_pipeline = pickle.load(f)

with open("checkerboard.pkl", "rb") as f:
    cb = pickle.load(f)

print("Loaded SfM pipeline and checkerboard data")

# ------------------------------
# Load SfM reconstruction
# ------------------------------
sparse_path = "sparse/0"
rec = pycolmap.Reconstruction(sparse_path)
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)

# Get SfM poses
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()
print(f"SfM poses: {len(sfm_rotations)} rotations, {len(sfm_translations)} translations")

# Get checkerboard poses (relative to checkerboard)
checker_rotations, checker_translations = cb.get_poses()
print(f"Checkerboard poses: {len(checker_rotations)} rotations, {len(checker_translations)} translations")

# ------------------------------
# Transform checkerboard poses to satellite frame
# ------------------------------
# Define checkerboard → satellite transform (example)
R_cb_to_sat = np.array([
    [ 0.96917272, 0.0, 0.24638229],
    [ 0.0,        1.0, 0.0      ],
    [-0.24638229, 0.0, 0.96917272]
])
t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]]) / 1000.0  # mm → m

T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3, :3] = R_cb_to_sat
T_cb_to_sat[:3, 3] = t_cb_to_sat.flatten()

T_sat_to_cam_list = []
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    T_cb_to_cam = np.eye(4)
    T_cb_to_cam[:3, :3] = R_cb
    T_cb_to_cam[:3, 3] = t_cb.flatten()
    
    # Transform camera to satellite frame
    T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
    T_sat_to_cam_list.append(T_sat_to_cam)

# ------------------------------
# Match images between SfM and checkerboard
# ------------------------------
cb_images = cb._checker_image_names
matched_indices_sfm = []
matched_indices_cb = []

for j, cb_name in enumerate(cb_images):
    for i, sfm_img in enumerate(sfm_images):
        if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
            matched_indices_sfm.append(i)
            matched_indices_cb.append(j)
            break

# Filter poses to only matched images
sfm_rotations_matched = [sfm_rotations[i] for i in matched_indices_sfm]
sfm_translations_matched = [sfm_translations[i] for i in matched_indices_sfm]
T_sat_to_cam_matched = [T_sat_to_cam_list[j] for j in matched_indices_cb]

print(f"Matched {len(sfm_rotations_matched)} SfM poses with checkerboard poses")


# ------------------------------
# Compute scale from matched translations
# ------------------------------
sfm_t_array = np.hstack([t for t in sfm_translations_matched])           # 3 x N
cb_t_array  = np.hstack([T[:3,3].reshape(3,1) for T in T_sat_to_cam_matched])  # 3 x N

# Center translations
sfm_t_centered = sfm_t_array - sfm_t_array[:, [0]]
cb_t_centered  = cb_t_array - cb_t_array[:, [0]]

# Scale factor
s = np.sum(cb_t_centered * sfm_t_centered) / np.sum(sfm_t_centered**2)
print(f"Estimated scale factor: {s:.6f}")

# Apply scale to SfM translations
sfm_translations_scaled = [s * t for t in sfm_translations]

# ------------------------------
# Optional: Compute rotation alignment
# ------------------------------
# If SfM axes are rotated relative to satellite frame
def compute_rotation_alignment(S, C):
    """
    Compute optimal rotation aligning SfM translations S to checkerboard translations C
    using Kabsch algorithm.
    S, C: 3 x N arrays (centered)
    Returns 3x3 rotation matrix
    """
    H = S @ C.T
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        # Reflection correction
        Vt[2,:] *= -1
        R_align = Vt.T @ U.T
    return R_align

R_align = compute_rotation_alignment(sfm_t_centered, cb_t_centered)
sfm_rotations_aligned = [R_align @ R for R in sfm_rotations_matched]
sfm_translations_aligned = [R_align @ t for t in sfm_translations_scaled]

# ------------------------------
# Compute errors
# ------------------------------
def rotation_error(R1, R2):
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

rotation_diffs = []
translation_diffs = []
for i in range(len(sfm_rotations_aligned)):
    R_sfm = sfm_rotations_aligned[i]
    t_sfm = sfm_translations_aligned[i]
    T_cb = T_sat_to_cam_matched[i]
    R_cb = T_cb[:3,:3]
    t_cb = T_cb[:3,3]

    rotation_diffs.append(rotation_error(R_sfm, R_cb))
    translation_diffs.append(translation_error(t_sfm, t_cb))

print("\nPose validation results:")
for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Image {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")

























# import numpy as np
# import open3d as o3d
# import os

# # Define transformation from checkerboard to satellite frame
# # R_cb_to_sat = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.0, 0.0])
# # t_cb_to_sat = np.array([[0.05], [0.02], [0.01]])

# R_cb_to_sat = np.array([
#     [ 0.96917272, 0.0, 0.24638229],
#     [ 0.0,        1.0, 0.0      ],
#     [-0.24638229, 0.0, 0.96917272]
# ])
# t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]])
# t_cb_to_sat = t_cb_to_sat / 1000.0  # if SolidWorks outputs mm


# # Build homogeneous transform
# T_cb_to_sat = np.eye(4)
# T_cb_to_sat[:3, :3] = R_cb_to_sat
# T_cb_to_sat[:3, 3:] = t_cb_to_sat

# # Convert checkerboard poses to satellite frame
# T_sat_to_cam_list = []
# for R_cb, t_cb in zip(checker_rotations, checker_translations):
#     T_cb_to_cam = np.eye(4)
#     T_cb_to_cam[:3, :3] = R_cb
#     T_cb_to_cam[:3, 3:] = t_cb
#     # satellite → cam = cb → cam × sat → cb⁻¹
#     T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
#     T_sat_to_cam_list.append(T_sat_to_cam)

# # ---------------------------------------------------------
# # Align SfM and Checkerboard poses by image names
# # ---------------------------------------------------------
# rec_path = os.path.join(sfm_pipeline._sparse_path, "0")
# rec = pycolmap.Reconstruction(rec_path)
# sfm_images = list(rec.images.values())  # SfM images

# cb_images = cb._checker_image_names  # checkerboard-detected image filenames

# matched_indices_sfm = []
# matched_indices_cb = []

# for j, cb_name in enumerate(cb_images):
#     for i, sfm_img in enumerate(sfm_images):
#         if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
#             matched_indices_sfm.append(i)
#             matched_indices_cb.append(j)
#             break

# # Filter both datasets to only matched images
# sfm_rotations = [sfm_rotations[i] for i in matched_indices_sfm]
# sfm_translations = [sfm_translations[i] for i in matched_indices_sfm]
# T_sat_to_cam_list = [T_sat_to_cam_list[j] for j in matched_indices_cb]

# print(f"Matched {len(sfm_rotations)} SfM poses with checkerboard detections.")

# # Debugging
# print("SfM camera centers (m):")
# for t in sfm_translations:
#     print(t.flatten())

# print("Checkerboard camera centers (m):")
# for T in T_sat_to_cam_list:
#     print(T[:3,3].flatten())


# # ---------------------------------------------------------
# # Compute and Apply Scale
# # ---------------------------------------------------------
# # sfm_t_array = np.hstack([t for t in sfm_translations])               # 3 x N
# # cb_t_array  = np.hstack([T[:3, 3:] for T in T_sat_to_cam_list])      # 3 x N

# # s = np.sum(cb_t_array * sfm_t_array) / np.sum(sfm_t_array**2)
# # print(f"Estimated scale factor: {s:.4f}")

# # sfm_translations_scaled = [s * t for t in sfm_translations]


# sfm_t_array = np.hstack([t for t in sfm_translations])           # 3 x N
# cb_t_array  = np.hstack([T[:3, 3:] for T in T_sat_to_cam_list])  # 3 x N

# # Center around first camera to remove origin offset
# sfm_t_array_centered = sfm_t_array - sfm_t_array[:, [0]]
# cb_t_array_centered  = cb_t_array - cb_t_array[:, [0]]

# # Compute scale factor using centered translations
# s = np.sum(cb_t_array_centered * sfm_t_array_centered) / np.sum(sfm_t_array_centered**2)

# # After computing R_align, t_align, and s
# print(f"Estimated scale factor: {s}")

# # --- Fix for negative scale or reflection ---
# if s < 0:
#     print("Negative scale detected — flipping coordinate system for consistency.")
#     s = -s
#     R_cb_to_sat[:, 2] *= -1  # Flip z-axis to correct handedness/reflection


# sfm_translations_scaled = [s * t for t in sfm_translations]

# # ---------------------------------------------------------
# # 4. Compute differences between SfM and checkerboard-derived poses
# # ---------------------------------------------------------
# def rotation_error(R1, R2):
#     """Return angular difference (deg) between two rotation matrices."""
#     dR = R1.T @ R2
#     angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
#     return np.degrees(angle)

# def translation_error(t1, t2):
#     """Return Euclidean distance (m) between two translation vectors."""
#     return np.linalg.norm(t1 - t2)

# rotation_diffs = []
# translation_diffs = []
# num_images = len(T_sat_to_cam_list)  # matches SfM-used images

# for i in range(num_images):
#     R_sfm, t_sfm = sfm_rotations[i], sfm_translations_scaled[i]
#     T_sat_cam_est = T_sat_to_cam_list[i]

#     R_sat_cam = T_sat_cam_est[:3, :3]
#     t_sat_cam = T_sat_cam_est[:3, 3:]

#     rot_err = rotation_error(R_sfm, R_sat_cam)
#     trans_err = translation_error(t_sfm, t_sat_cam)

#     rotation_diffs.append(rot_err)
#     translation_diffs.append(trans_err)

# # ---------------------------------------------------------
# # 5. Report validation results
# # ---------------------------------------------------------
# print("Pose validation results:")
# for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
#     print(f"Image {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

# print(f"\nMean rotation error: {np.mean(rotation_diffs):.3f}°")
# print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")








import os
import numpy as np
import open3d as o3d
import pycolmap
from scipy.spatial.transform import Rotation

# -----------------------------
# 1. Load scaled SfM point cloud
# -----------------------------
sparse_path = "sparse/0"
pcd = o3d.io.read_point_cloud(os.path.join(sparse_path, "points.ply"))
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# -----------------------------
# 2. Load SfM reconstruction
# -----------------------------
rec = pycolmap.Reconstruction(sparse_path)
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)
sfm_cameras = {cam_id: cam for cam_id, cam in rec.cameras.items()}

# Example: SfM rotations/translations
sfm_rotations = [img.cam_from_world().rotation.matrix() for img in sfm_images]
sfm_translations = [img.projection_center().flatten() for img in sfm_images]  # 3, 

# -----------------------------
# 3. Checkerboard camera poses
# -----------------------------
# T_cb_to_sat is your checkerboard → satellite transform
# Example:
R_cb_to_sat = np.array([
    [ 0.96917272, 0.0, 0.24638229],
    [ 0.0,        1.0, 0.0      ],
    [-0.24638229, 0.0, 0.96917272]
])
t_cb_to_sat = np.array([27.41, 0.0, -35.25]) / 1000.0

T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3,:3] = R_cb_to_sat
T_cb_to_sat[:3,3]  = t_cb_to_sat

# Suppose you have checker_rotations / checker_translations arrays
T_sat_to_cam_list = []
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    T_cb_to_cam = np.eye(4)
    T_cb_to_cam[:3,:3] = R_cb
    T_cb_to_cam[:3,3]  = t_cb.flatten()
    T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
    T_sat_to_cam_list.append(T_sat_to_cam)

# -----------------------------
# 4. Match images by name
# -----------------------------
cb_images = cb._checker_image_names
matched_indices_sfm = []
matched_indices_cb = []

for j, cb_name in enumerate(cb_images):
    for i, sfm_img in enumerate(sfm_images):
        if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
            matched_indices_sfm.append(i)
            matched_indices_cb.append(j)
            break

sfm_rotations_matched = [sfm_rotations[i] for i in matched_indices_sfm]
sfm_translations_matched = [sfm_translations[i] for i in matched_indices_sfm]
T_sat_to_cam_list_matched = [T_sat_to_cam_list[j] for j in matched_indices_cb]

# -----------------------------
# 5. Compute scale & rotation alignment (Kabsch)
# -----------------------------
sfm_pts = np.stack(sfm_translations_matched, axis=1)  # 3 x N
cb_pts  = np.stack([T[:3,3] for T in T_sat_to_cam_list_matched], axis=1)

# Center points
sfm_mean = sfm_pts.mean(axis=1, keepdims=True)
cb_mean  = cb_pts.mean(axis=1, keepdims=True)
sfm_centered = sfm_pts - sfm_mean
cb_centered  = cb_pts - cb_mean

# Scale
s = np.sum(cb_centered * sfm_centered) / np.sum(sfm_centered**2)
sfm_centered_scaled = s * sfm_centered

# Rotation (Kabsch)
H = sfm_centered_scaled @ cb_centered.T
U, _, Vt = np.linalg.svd(H)
R_align = Vt.T @ U.T

# Handle reflection
if np.linalg.det(R_align) < 0:
    Vt[2,:] *= -1
    R_align = Vt.T @ U.T

# Apply scale + rotation
sfm_translations_aligned = [ (R_align @ (s*t.flatten())).flatten() for t in sfm_translations_matched ]
sfm_rotations_aligned = [ R_align @ R for R in sfm_rotations_matched ]

# -----------------------------
# 6. Compute rotation/translation errors
# -----------------------------
def rotation_error(R1,R2):
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR)-1)/2,-1,1))
    return np.degrees(angle)

def translation_error(t1,t2):
    return np.linalg.norm(t1-t2)

rotation_diffs=[]
translation_diffs=[]
for i in range(len(T_sat_to_cam_list_matched)):
    R_cb = T_sat_to_cam_list_matched[i][:3,:3]
    t_cb = T_sat_to_cam_list_matched[i][:3,3]
    rotation_diffs.append(rotation_error(sfm_rotations_aligned[i], R_cb))
    translation_diffs.append(translation_error(sfm_translations_aligned[i], t_cb))

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")

# -----------------------------
# 7. Visualization
# -----------------------------
def make_camera_frustum(R, t, scale=0.05, color=[1,0,0]):
    corners = np.array([
        [0,0,0],
        [-0.5,-0.5,1.0],
        [0.5,-0.5,1.0],
        [0.5,0.5,1.0],
        [-0.5,0.5,1.0]
    ]) * scale
    corners_world = (R @ corners.T).T + t
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return line_set

# Point cloud scaled
pcd_scaled = o3d.geometry.PointCloud()
pcd_scaled.points = o3d.utility.Vector3dVector(points * s)
pcd_scaled.colors = o3d.utility.Vector3dVector(colors)

# SfM cameras
geometries_sfm = [pcd_scaled]
for R,t in zip(sfm_rotations_aligned,sfm_translations_aligned):
    geometries_sfm.append(make_camera_frustum(R,t,color=[1,0,0]))

o3d.visualization.draw_geometries(geometries_sfm, window_name="SfM Aligned (Red)")

# Checkerboard cameras
geometries_cb = [pcd_scaled]
for T in T_sat_to_cam_list_matched:
    geometries_cb.append(make_camera_frustum(T[:3,:3], T[:3,3], color=[0,1,0]))

o3d.visualization.draw_geometries(geometries_cb, window_name="Checkerboard Cameras (Green)")
































# Original call
# plot_pointcloud(
#     sparse_path="sparse",
#     store_path="0",
#     camera_scale=0.1,
#     matched_indices=matched_indices_sfm  # Only plot the SfM cameras matched to checkerboard
# )



# plot_pointcloud(
#     sparse_path="sparse",
#     store_path="0",
#     camera_scale=0.1,
#     matched_indices=matched_indices_sfm,
#     translations_scaled=sfm_translations_scaled,
#     scale_pointcloud=s
# )

plot_pointcloud("sparse", store_path="0", camera_scale=0.1, matched_indices=matched_indices_sfm)







pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
points = np.asarray(pcd.points) * s
pcd_scaled = o3d.geometry.PointCloud()
pcd_scaled.points = o3d.utility.Vector3dVector(points)
pcd_scaled.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))

plot_pointcloud_checkerboard(pcd_scaled, T_sat_to_cam_list, frustum_scale=0.05)











# import open3d as o3d
# import pycolmap
# import numpy as np
# import os

# # ---------------------------------------------------------
# # 1. Load and scale SfM point cloud
# # ---------------------------------------------------------
# print("=== Loading and scaling SfM point cloud ===")
# sfm_path = "sparse/0"
# pcd = o3d.io.read_point_cloud(os.path.join(sfm_path, "points.ply"))

# points = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors)
# points_scaled = points * s  # Apply scale factor

# scaled_pcd = o3d.geometry.PointCloud()
# scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
# scaled_pcd.colors = o3d.utility.Vector3dVector(colors)


# # ---------------------------------------------------------
# # 2. Visualize raw SfM reconstruction (world frame)
# # ---------------------------------------------------------
# print("\n=== Visualizing original SfM reconstruction ===")
# plot_pointcloud(
#     sparse_path="sparse",     # Path to parent folder
#     store_path="0",           # Subfolder name
#     camera_scale=0.1          # Frustum/axis scale
# )


# # ---------------------------------------------------------
# # 3. Build transformed point cloud (aligned to checkerboard frame)
# # ---------------------------------------------------------
# print("\n=== Transforming and visualizing scaled SfM point cloud in checkerboard/satellite frame ===")

# # Build transformation matrices for each checkerboard pose
# T_cb_to_sat = np.eye(4)
# T_cb_to_sat[:3, :3] = R_cb_to_sat
# T_cb_to_sat[:3, 3:] = t_cb_to_sat

# # Transform scaled point cloud from SfM frame to satellite frame
# # (Optional depending on how you define your alignment)
# points_cb_frame = (R_cb_to_sat @ points_scaled.T).T + t_cb_to_sat.flatten()

# cb_pcd = o3d.geometry.PointCloud()
# cb_pcd.points = o3d.utility.Vector3dVector(points_cb_frame)
# cb_pcd.colors = o3d.utility.Vector3dVector(colors)

# # Save optional transformed cloud
# o3d.io.write_point_cloud("checkerboard_frame_points.ply", cb_pcd)

# # # ---------------------------------------------------------
# # # 4. Visualize transformed (checkerboard-aligned) reconstruction
# # # ---------------------------------------------------------
# # o3d.visualization.draw_geometries(
# #     [cb_pcd],
# #     window_name="Scaled Point Cloud in Checkerboard/Satellite Frame",
# #     width=1280,
# #     height=800,
# # )


# # ---------------------------------------------------------
# # 3. Add checkerboard camera poses as coordinate frames
# # ---------------------------------------------------------
# camera_scale = 0.1  # size of coordinate frames
# camera_frames = []

# for T_sat_cam in T_sat_to_cam_list:
#     R_cb = T_sat_cam[:3, :3]
#     t_cb = T_sat_cam[:3, 3].flatten()

#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#     T_frame = np.eye(4)
#     T_frame[:3, :3] = R_cb
#     T_frame[:3, 3] = t_cb
#     frame.transform(T_frame)

#     camera_frames.append(frame)

# # Visualize point cloud with checkerboard camera poses
# o3d.visualization.draw_geometries(
#     [cb_pcd, *camera_frames],
#     window_name="Checkerboard Cameras + Scaled Point Cloud",
#     width=1280,
#     height=800
# )












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
camera_scale = 0.1
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
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_lines)
    return line_set

# ---------------------------------------------------------
# Load SfM reconstruction
# ---------------------------------------------------------
rec = pycolmap.Reconstruction("sparse/0")
sfm_images = sorted(rec.images.values(), key=lambda x: x.name)
sfm_cameras = {cam_id: cam for cam_id, cam in rec.cameras.items()}


for i in range(len(sfm_translations_scaled)):
    img = sfm_images[matched_indices_sfm[i]]  # get the corresponding SfM image
    t = sfm_translations_scaled[i].flatten()  # already matched & scaled

    R = img.cam_from_world().rotation.matrix()

    cam = sfm_cameras[img.camera_id]
    fx = cam.params[0] if len(cam.params) > 0 else 1
    fy = cam.params[1] if len(cam.params) > 1 else fx
    width, height = cam.width, cam.height

    ls_sfm = make_camera_frustum(R, t, width, height, fx, fy, color=[1,0,0])
    geometries.append(ls_sfm)

    # # Checkerboard camera
    # T_cb = T_sat_to_cam_list[i]
    # R_cb = T_cb[:3,:3]
    # t_cb = T_cb[:3,3]
    # ls_cb = make_camera_frustum(R_cb, t_cb, color=[0,1,0])
    # geometries.append(ls_cb)

# ---------------------------------------------------------
# Visualize everything
# ---------------------------------------------------------
o3d.visualization.draw_geometries(
    geometries,
    window_name="Scaled Point Cloud + Camera Poses (SfM=Red, Checkerboard=Green)"
)





















# # ---------------------------------------------------------
# # Load scaled SfM point cloud
# # ---------------------------------------------------------
# print("Load and scale point cloud")
# pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
# points = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors)
# points_scaled = points * s  # apply scale factor

# scaled_pcd = o3d.geometry.PointCloud()
# scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
# scaled_pcd.colors = o3d.utility.Vector3dVector(colors)

# # Prepare for visualization
# geometries = [scaled_pcd]

# # ---------------------------------------------------------
# # Helper: Create camera frustum from real intrinsics
# # ---------------------------------------------------------
# def make_camera_frustum(R, t, width, height, fx, fy, scale=0.1, color=[1, 0, 0]):
#     """Create an Open3D LineSet for a camera frustum using real intrinsics."""
#     cx = width / 2
#     cy = height / 2
#     frustum_depth = scale * 2

#     # Frustum corners in camera coordinates
#     corners_cam = np.array([
#         [0, 0, 0],  # Camera center
#         [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#         [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#         [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#         [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#     ])

#     # Transform frustum corners to world coordinates
#     corners_world = (R.T @ corners_cam.T).T + t

#     # Define edges of the frustum pyramid
#     lines = [[0, 1], [0, 2], [0, 3], [0, 4],
#              [1, 2], [2, 3], [3, 4], [4, 1]]

#     # Build LineSet
#     ls = o3d.geometry.LineSet()
#     ls.points = o3d.utility.Vector3dVector(corners_world)
#     ls.lines = o3d.utility.Vector2iVector(lines)
#     ls.colors = o3d.utility.Vector3dVector([color for _ in lines])

#     return ls


# # ---------------------------------------------------------
# # Load SfM reconstruction (for camera intrinsics & poses)
# # ---------------------------------------------------------
# rec = pycolmap.Reconstruction("sparse/0")
# sfm_images = sorted(rec.images.values(), key=lambda x: x.name)
# sfm_cameras = {cam_id: cam for cam_id, cam in rec.cameras.items()}

# camera_scale = 0.1  # for visual size

# # ---------------------------------------------------------
# # Plot SfM and Checkerboard camera frustums
# # ---------------------------------------------------------
# for i in range(len(sfm_translations_scaled)):
#     # --- SfM camera ---
#     img = sfm_images[matched_indices_sfm[i]]
#     R_sfm = img.cam_from_world().rotation.matrix()
#     t_sfm = sfm_translations_scaled[i].flatten()  # already scaled

#     cam = sfm_cameras[img.camera_id]
#     width, height = cam.width, cam.height
#     params = cam.params
#     if len(params) >= 2:
#         fx, fy = params[0], params[1]
#     else:
#         fx = fy = params[0] if len(params) > 0 else width

#     ls_sfm = make_camera_frustum(R_sfm, t_sfm, width, height, fx, fy,
#                                  scale=camera_scale, color=[1, 0, 0])
#     geometries.append(ls_sfm)

#     # --- Checkerboard camera ---
#     T_cb = T_sat_to_cam_list[i]
#     R_cb = T_cb[:3, :3]
#     t_cb = T_cb[:3, 3]
#     ls_cb = make_camera_frustum(R_cb, t_cb, width, height, fx, fy,
#                                 scale=camera_scale, color=[0, 1, 0])
#     geometries.append(ls_cb)

# # ---------------------------------------------------------
# # Visualize everything
# # ---------------------------------------------------------
# print(f"Visualizing {len(sfm_translations_scaled)} matched cameras")
# o3d.visualization.draw_geometries(
#     geometries,
#     window_name="Scaled Point Cloud + Camera Poses (SfM=Red, Checkerboard=Green)",
#     width=1280,
#     height=800,
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

