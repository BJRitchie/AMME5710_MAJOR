import pycolmap
import matplotlib.pyplot as plt

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard

# Convert the video into images 
# vid_path = "images/batmo.mp4"
# vid_path = 'images/ben.mp4'
# store_path="images/ben"
vid_path = 'images/hanging_sat_checkerboard.mp4'
store_path="images/hanging_sat_checkerboard"
gen_images_from_vid( vid_path, store_path ) 

# Storage files 
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"

# Settings 
sift_ops = pycolmap.SiftExtractionOptions()
sift_ops.use_gpu = False # CPU only 
sift_ops.first_octave = 0
sift_ops.num_octaves = 4

# Initialise the pipeline 
sfm_pipeline = pipeline.StrcFromMotion ( 
    db_path, im_path, sparse_path, dense_path,
    cam_mode    =pycolmap.CameraMode.AUTO, 
    cam_model   ="SIMPLE_RADIAL",  
    reader_ops  =pycolmap.ImageReaderOptions(), 
    sift_ops    =sift_ops, 
    device      =pycolmap.Device.cpu 
) 

sfm_pipeline.resize_ims( store_path, 1200, 5 )  # originally 10 but wasn't using images with checkerboard in it
sfm_pipeline.prep_pointcloud() 
sfm_pipeline.make_pointcloud()
sfm_pipeline.clean_pointcloud() # TODO experiment with this a bit more - gets rid of far outliers but not close ones (just off of body)
sfm_pipeline.plot_pointcloud() 


# Save only the images that SfM used
sfm_pipeline.save_registered_images(output_folder="images/hanging_sat_checkerboard_sfm")

# Get SFM pose estimates
sfm_rotations, sfm_translations = sfm_pipeline.get_poses()

print(len(sfm_rotations))
print(len(sfm_translations))

# Checkerboard detection on images the SFM used 
cb = checkerboard.Checkerboard() 
cb.read_ims("images/hanging_sat_checkerboard_sfm") 
cb.undistort_ims(grid_size=(3, 3), cell_size=0.096)
cb.plot_checkerboards() 

plt.show()

# Extract checkerboard pose estimates 
checker_rotations, checker_translations = cb.get_poses()



import numpy as np
import open3d as o3d

# ----------------------------
# Convert checkerboard poses to satellite frame
# ----------------------------
R_cb_to_sat = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.0, 0.0])
t_cb_to_sat = np.array([[0.05], [0.02], [0.01]])

# Build homogeneous transform
T_cb_to_sat = np.eye(4)
T_cb_to_sat[:3, :3] = R_cb_to_sat
T_cb_to_sat[:3, 3:] = t_cb_to_sat

# ---------------------------------------------------------
# Convert checkerboard poses to satellite frame
# ---------------------------------------------------------
T_sat_to_cam_list = []
for R_cb, t_cb in zip(checker_rotations, checker_translations):
    T_cb_to_cam = np.eye(4)
    T_cb_to_cam[:3, :3] = R_cb
    T_cb_to_cam[:3, 3:] = t_cb
    # satellite → cam = cb → cam × sat → cb⁻¹
    T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
    T_sat_to_cam_list.append(T_sat_to_cam)






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
    R_sfm, t_sfm = sfm_rotations[i], sfm_translations[i]
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
# 6. Optional: visualize camera poses for comparison
# ---------------------------------------------------------
def make_frame(T, color):
    """Create Open3D coordinate frame geometry from a 4x4 pose matrix."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    frame.paint_uniform_color(color)
    frame.transform(T)
    return frame

sfm_frames = [make_frame(np.vstack([np.hstack([R, t]), [0, 0, 0, 1]]), [1, 0, 0]) for R, t in zip(sfm_rotations, sfm_translations)]
cb_frames = [make_frame(T, [0, 1, 0]) for T in T_sat_to_cam_list]

o3d.visualization.draw_geometries(sfm_frames + cb_frames, window_name="Pose Comparison (SfM=Red, Checkerboard=Green)")


