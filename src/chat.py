import numpy as np
import pycolmap
import pickle
import os

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
# Compute rotation and translation errors (no scaling)
# ------------------------------
def rotation_error(R1, R2):
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

rotation_diffs = []
translation_diffs = []

for i in range(len(sfm_rotations_matched)):
    R_sfm = sfm_rotations_matched[i]
    t_sfm = sfm_translations_matched[i]
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




print("\n\n")


# ------------------------------
# Compute scale from matched translations (without touching rotation)
# ------------------------------
sfm_t_array = np.hstack([t for t in sfm_translations_matched])           # 3 x N
cb_t_array  = np.hstack([T[:3,3].reshape(3,1) for T in T_sat_to_cam_matched])  # 3 x N

# Center translations to remove origin offset
sfm_t_centered = sfm_t_array - sfm_t_array[:, [0]]
cb_t_centered  = cb_t_array - cb_t_array[:, [0]]

# Compute scale factor
s = np.sum(cb_t_centered * sfm_t_centered) / np.sum(sfm_t_centered**2)
print(f"Estimated scale factor: {s:.6f}")

# Apply scale to SfM translations
sfm_translations_scaled = [s * t for t in sfm_translations_matched]

# ------------------------------
# Compute translation and rotation errors
# ------------------------------
def rotation_error(R1, R2):
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1.0, 1.0))
    return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

rotation_diffs = []
translation_diffs = []
for i in range(len(sfm_rotations_matched)):
    R_sfm = sfm_rotations_matched[i]      # unchanged
    t_sfm = sfm_translations_scaled[i]    # scaled
    T_cb = T_sat_to_cam_matched[i]
    R_cb = T_cb[:3,:3]
    t_cb = T_cb[:3,3]

    rotation_diffs.append(rotation_error(R_sfm, R_cb))
    translation_diffs.append(translation_error(t_sfm, t_cb))

print("\nPose validation results (scale applied only):")
for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Camera {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")







# import numpy as np
# import pycolmap
# import pickle
# import os

# # ------------------------------
# # Load SfM pipeline and checkerboard data
# # ------------------------------
# with open("sfm_pipeline.pkl", "rb") as f:
#     sfm_pipeline = pickle.load(f)

# with open("checkerboard.pkl", "rb") as f:
#     cb = pickle.load(f)

# print("Loaded SfM pipeline and checkerboard data")

# # ------------------------------
# # Load SfM reconstruction
# # ------------------------------
# sparse_path = "sparse/0"
# rec = pycolmap.Reconstruction(sparse_path)
# sfm_images = sorted(rec.images.values(), key=lambda x: x.name)

# # Get SfM poses
# sfm_rotations, sfm_translations = sfm_pipeline.get_poses()
# print(f"SfM poses: {len(sfm_rotations)} rotations, {len(sfm_translations)} translations")

# # Get checkerboard poses (relative to checkerboard)
# checker_rotations, checker_translations = cb.get_poses()
# print(f"Checkerboard poses: {len(checker_rotations)} rotations, {len(checker_translations)} translations")

# # ------------------------------
# # Transform checkerboard poses to satellite frame
# # ------------------------------
# R_cb_to_sat = np.array([
#     [ 0.96917272, 0.0, 0.24638229],
#     [ 0.0,        1.0, 0.0      ],
#     [-0.24638229, 0.0, 0.96917272]
# ])
# t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]]) / 1000.0  # mm → m

# T_cb_to_sat = np.eye(4)
# T_cb_to_sat[:3, :3] = R_cb_to_sat
# T_cb_to_sat[:3, 3] = t_cb_to_sat.flatten()

# # Transform checkerboard poses to satellite frame
# T_sat_to_cam_list = []
# for R_cb, t_cb in zip(checker_rotations, checker_translations):
#     T_cb_to_cam = np.eye(4)
#     T_cb_to_cam[:3, :3] = R_cb
#     T_cb_to_cam[:3, 3] = t_cb.flatten()
#     T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
#     T_sat_to_cam_list.append(T_sat_to_cam)

# # ------------------------------
# # Match images between SfM and checkerboard
# # ------------------------------
# cb_images = cb._checker_image_names
# matched_indices_sfm = []
# matched_indices_cb = []

# for j, cb_name in enumerate(cb_images):
#     for i, sfm_img in enumerate(sfm_images):
#         if os.path.basename(sfm_img.name) == os.path.basename(cb_name):
#             matched_indices_sfm.append(i)
#             matched_indices_cb.append(j)
#             break

# # Filter poses to only matched images
# sfm_rotations_matched = [sfm_rotations[i] for i in matched_indices_sfm]
# sfm_translations_matched = [sfm_translations[i] for i in matched_indices_sfm]
# T_sat_to_cam_matched = [T_sat_to_cam_list[j] for j in matched_indices_cb]

# print(f"Matched {len(sfm_rotations_matched)} SfM poses with checkerboard poses")

# # ------------------------------
# # Compute scale from matched translations
# # ------------------------------
# sfm_t_array = np.hstack([t for t in sfm_translations_matched])           # 3 x N
# cb_t_array  = np.hstack([T[:3,3].reshape(3,1) for T in T_sat_to_cam_matched])  # 3 x N

# # Center translations around first camera
# sfm_t_centered = sfm_t_array - sfm_t_array[:, [0]]
# cb_t_centered  = cb_t_array - cb_t_array[:, [0]]

# # Compute scale factor
# s = np.sum(cb_t_centered * sfm_t_centered) / np.sum(sfm_t_centered**2)
# if s < 0:
#     print("Negative scale detected — flipping scale to be positive")
#     s = -s

# print(f"Estimated scale factor: {s:.6f}")

# # Apply scale to SfM translations
# sfm_translations_scaled = [s * t for t in sfm_translations_matched]

# # ------------------------------
# # Compute rotation alignment (Kabsch)
# # ------------------------------
# def compute_rotation_alignment(S, C):
#     H = S @ C.T
#     U, _, Vt = np.linalg.svd(H)
#     R_align = Vt.T @ U.T
#     if np.linalg.det(R_align) < 0:
#         # Reflection correction
#         Vt[2,:] *= -1
#         R_align = Vt.T @ U.T
#     return R_align

# R_align = compute_rotation_alignment(sfm_t_centered, cb_t_centered)
# sfm_rotations_aligned = [R_align @ R for R in sfm_rotations_matched]
# sfm_translations_aligned = [R_align @ t for t in sfm_translations_scaled]

# # ------------------------------
# # Compute rotation and translation errors
# # ------------------------------
# def rotation_error(R1, R2):
#     dR = R1.T @ R2
#     angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1.0, 1.0))
#     return np.degrees(angle)

# def translation_error(t1, t2):
#     return np.linalg.norm(t1 - t2)

# rotation_diffs = []
# translation_diffs = []

# for i in range(len(sfm_rotations_aligned)):
#     R_sfm = sfm_rotations_aligned[i]
#     t_sfm = sfm_translations_aligned[i]
#     T_cb = T_sat_to_cam_matched[i]
#     R_cb = T_cb[:3,:3]
#     t_cb = T_cb[:3,3]

#     rotation_diffs.append(rotation_error(R_sfm, R_cb))
#     translation_diffs.append(translation_error(t_sfm, t_cb))

# print("\nPose validation results:")
# for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
#     print(f"Image {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

# print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
# print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")
