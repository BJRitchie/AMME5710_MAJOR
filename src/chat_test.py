import numpy as np
import open3d as o3d

num_cameras = 5
checker_rotations = []
checker_translations = []

# ------------------------------
# Synthetic Checkerboard Poses
# ------------------------------
for i in range(num_cameras):
    # small random rotation (~1°)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-0.017, 0.017)  # radians ~1°
    c = np.cos(angle)
    s = np.sin(angle)
    t_ = 1 - c
    x, y, z = axis
    R = np.array([
        [t_*x*x + c,   t_*x*y - s*z, t_*x*z + s*y],
        [t_*x*y + s*z, t_*y*y + c,   t_*y*z - s*x],
        [t_*x*z - s*y, t_*y*z + s*x, t_*z*z + c]
    ])
    checker_rotations.append(R)

    # small translation (~cm)
    t_vec = np.array([[0.5 + np.random.uniform(-0.005, 0.005)],
                      [0.0 + np.random.uniform(-0.005, 0.005)],
                      [0.0 + np.random.uniform(-0.005, 0.005)]])
    checker_translations.append(t_vec)

# ------------------------------
# Synthetic SfM Poses (small perturbation)
# ------------------------------
sfm_rotations = []
sfm_translations = []

for R_cb, t_cb in zip(checker_rotations, checker_translations):
    # small rotation perturbation (~0.5°)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-0.0087, 0.0087)  # radians ~0.5°
    c = np.cos(angle)
    s = np.sin(angle)
    t_ = 1 - c
    x, y, z = axis
    R_pert = np.array([
        [t_*x*x + c,   t_*x*y - s*z, t_*x*z + s*y],
        [t_*x*y + s*z, t_*y*y + c,   t_*y*z - s*x],
        [t_*x*z - s*y, t_*y*z + s*x, t_*z*z + c]
    ])
    R_sfm = R_pert @ R_cb

    # small translation perturbation (~mm)
    t_sfm = t_cb + np.random.uniform(-0.002, 0.002, (3,1))
    sfm_rotations.append(R_sfm)
    sfm_translations.append(t_sfm)

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

# ------------------------------
# Compute rotation alignment (optional)
# ------------------------------
def compute_rotation_alignment(S, C):
    H = S @ C.T
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[2,:] *= -1
        R_align = Vt.T @ U.T
    return R_align

R_align = compute_rotation_alignment(np.hstack(sfm_translations_scaled), cb_t_array)
sfm_rotations_aligned = [R_align @ R for R in sfm_rotations]
sfm_translations_aligned = [R_align @ t for t in sfm_translations_scaled]

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
    rotation_diffs.append(rotation_error(sfm_rotations_aligned[i], checker_rotations[i]))
    translation_diffs.append(translation_error(sfm_translations_aligned[i], checker_translations[i]))

print("\nPose validation results:")
for i, (r_err, t_err) in enumerate(zip(rotation_diffs, translation_diffs)):
    print(f"Camera {i}: rotation error = {r_err:.3f}°, translation error = {t_err:.4f} m")

print(f"Mean rotation error: {np.mean(rotation_diffs):.3f}°")
print(f"Mean translation error: {np.mean(translation_diffs):.4f} m")











# ------------------------------
# Create a dummy point cloud
# ------------------------------
num_points = 50
points = np.random.uniform(-0.2, 0.2, (num_points, 3))
colors = np.ones((num_points, 3)) * 0.5  # grey
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# ------------------------------
# Helper to create camera frustum
# ------------------------------
def make_camera_frustum(R, t, scale=0.05, color=[1,0,0]):
    """
    Creates a small camera frustum as a LineSet.
    R: 3x3 rotation (world <- camera)
    t: 3x1 translation (camera center in world)
    """
    corners_cam = np.array([
        [0, 0, 0],          # camera origin
        [-0.5, -0.5, 1.0],  # image corners in camera frame
        [ 0.5, -0.5, 1.0],
        [ 0.5,  0.5, 1.0],
        [-0.5,  0.5, 1.0],
    ]) * scale

    corners_world = (R @ corners_cam.T).T + t.flatten()
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    colors = [color for _ in lines]

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum

# ------------------------------
# Visualize cameras
# ------------------------------
geometries = [pcd]

# SfM cameras (red)
for R_sfm, t_sfm in zip(sfm_rotations_aligned, sfm_translations_aligned):
    frustum = make_camera_frustum(R_sfm, t_sfm, scale=0.05, color=[1,0,0])
    geometries.append(frustum)

# Checkerboard cameras (green)
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    frustum = make_camera_frustum(R_cb, t_cb, scale=0.05, color=[0,1,0])
    geometries.append(frustum)

# ------------------------------
# Show visualization
# ------------------------------
o3d.visualization.draw_geometries(geometries, window_name="Synthetic SfM vs Checkerboard")