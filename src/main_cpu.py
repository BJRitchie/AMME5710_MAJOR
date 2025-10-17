# Generates point cloud

import pycolmap
import numpy as np
import cv2
import os
import shutil
import open3d as o3d
import matplotlib.pyplot as plt

################### CONFIGURATION ###################
# # ====> CHANGE THIS TO YOUR DATASET PATH
# DATASET_PATH = "/absolute/path/to/your/dataset"
# im_path = os.path.join(DATASET_PATH, "images")

# # Clean up any old files
# db_path = os.path.join(DATASET_PATH, "database.db")
# sparse_path = os.path.join(DATASET_PATH, "sparse")
# dense_path = os.path.join(DATASET_PATH, "dense")

# ====> CHANGE THIS TO YOUR DATASET PATH
im_path = "images/kelly"

# Clean up any old files
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"

for path in [db_path, sparse_path, dense_path]:
    if os.path.exists(path):
        print(f"Removing old file/folder: {path}")
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

# Recreate clean output directories
os.makedirs(sparse_path, exist_ok=True)
os.makedirs(dense_path, exist_ok=True)

################### IMAGE RESIZING ###################
im_names = [f for f in os.listdir(im_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
MAX_SIZE = 1200  # downscale largest dimension

for f in im_names:
    path = os.path.join(im_path, f)
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    scale = MAX_SIZE / max(h, w)
    if scale < 1.0:
        img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
        cv2.imwrite(path, img_resized)

################### INITIALISATION ###################
cam_mode = pycolmap.CameraMode.AUTO
cam_model = "SIMPLE_RADIAL"  # AUTO let COLMAP detect the camera model - AUTO DOESNT WORK

reader_ops = pycolmap.ImageReaderOptions()

sift_ops = pycolmap.SiftExtractionOptions()
sift_ops.use_gpu = False            # CPU only
sift_ops.first_octave = 0
sift_ops.num_octaves = 4

device = pycolmap.Device.cpu

################### FEATURE EXTRACTION ################
print("=== Extracting features ===")
pycolmap.extract_features(
    database_path=db_path,
    image_path=im_path,
    image_names=im_names,
    camera_mode=cam_mode,
    camera_model=cam_model,
    reader_options=reader_ops,
    sift_options=sift_ops,
    device=device,
)

################### FEATURE MATCHING ##################
print("=== Matching features (exhaustive) ===")
pycolmap.match_exhaustive(
    database_path=db_path,
    sift_options=pycolmap.SiftMatchingOptions(),
    matching_options=pycolmap.ExhaustiveMatchingOptions(),
    verification_options=pycolmap.TwoViewGeometryOptions(),
    device=device,
)

################### INCREMENTAL MAPPING ###############
print("=== Running incremental mapping ===")
pycolmap.incremental_mapping(
    database_path=db_path,
    image_path=im_path,
    output_path=sparse_path,
)

################### UNDISTORT IMAGES ##################
print("=== Undistorting images ===")
pycolmap.undistort_images(
    output_path=dense_path,
    input_path=os.path.join(sparse_path, "0"),  # first reconstruction
    image_path=im_path,
    output_type="COLMAP",
)

################### EXPORT POINT CLOUD ################
rec_path = os.path.join(sparse_path, "0")
rec = pycolmap.Reconstruction(rec_path)
print("Registered images:", len(rec.images))
print("3D points:", len(rec.points3D))

points_ply = os.path.join(rec_path, "points.ply")
rec.export_PLY(points_ply)
print("Saved sparse point cloud to", points_ply)

################### VISUALISATION #####################
if os.path.exists(points_ply):
    print("=== Loading and visualizing sparse point cloud ===")
    pcd = o3d.io.read_point_cloud(points_ply)
    o3d.visualization.draw_geometries([pcd])
else:
    print(f"No sparse point cloud found at {points_ply}. Sparse mapping might have failed.")

################### OPTIONAL: visualize first image with OpenCV SIFT ################
first_image_path = os.path.join(im_path, im_names[0])
img = cv2.imread(first_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
img_kp = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.figure(figsize=(10,8))
# plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
# plt.title("First image SIFT keypoints (CPU)")
# plt.axis('off')
# plt.show()













# # Generates point cloud - but stale point cloud remains

# import pycolmap
# import numpy as np
# import cv2
# import os
# import open3d as o3d
# import matplotlib.pyplot as plt

# import cv2
# import os

# DATASET_PATH = ""
# # im_path = os.path.join(DATASET_PATH, "images/south-building/images")
# im_path = os.path.join(DATASET_PATH, "images")

# im_names = [f for f in os.listdir(im_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# MAX_SIZE = 1200  # downscale largest dimension

# for f in im_names:
#     path = os.path.join(im_path, f)
#     img = cv2.imread(path)
#     h, w = img.shape[:2]
#     scale = MAX_SIZE / max(h, w)
#     if scale < 1.0:
#         img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
#         cv2.imwrite(path, img_resized)



# ################### INITIALISATION ###################
# DATASET_PATH = ""  # <-- Set your dataset path here
# db_path = os.path.join(DATASET_PATH, "database.db")
# # im_path = os.path.join(DATASET_PATH, "images/south-building/images")
# # im_path = os.path.join(DATASET_PATH, "images")


# # List image files
# im_names = [f for f in os.listdir(im_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# # Camera and feature extraction settings
# cam_mode = pycolmap.CameraMode.AUTO
# cam_model = "SIMPLE_RADIAL"

# reader_ops = pycolmap.ImageReaderOptions()

# # CPU-friendly SIFT options
# sift_ops = pycolmap.SiftExtractionOptions()
# sift_ops.use_gpu = False            # CPU only
# # sift_ops.max_num_threads = 1        # reduce threads to save RAM
# # sift_ops.max_image_size = 1200      # downscale large images
# sift_ops.first_octave = 0           # optional, reduce memory
# sift_ops.num_octaves = 4            # default

# device = pycolmap.Device.cpu         # force CPU

# ################### FEATURE EXTRACTION ################
# print("=== Extracting features ===")
# pycolmap.extract_features(
#     database_path=db_path,
#     image_path=im_path,
#     image_names=im_names,
#     camera_mode=cam_mode,
#     camera_model=cam_model,
#     reader_options=reader_ops,
#     sift_options=sift_ops,
#     device=device,
# )

# ################### FEATURE MATCHING ##################
# print("=== Matching features (exhaustive) ===")
# pycolmap.match_exhaustive(
#     database_path=db_path,
#     sift_options=pycolmap.SiftMatchingOptions(),
#     matching_options=pycolmap.ExhaustiveMatchingOptions(),
#     verification_options=pycolmap.TwoViewGeometryOptions(),
#     device=device,
# )

# ################### INCREMENTAL MAPPING ###############
# print("=== Running incremental mapping ===")
# inc_map_out_path = os.path.join(DATASET_PATH, "sparse")
# os.makedirs(inc_map_out_path, exist_ok=True)

# pycolmap.incremental_mapping(
#     database_path=db_path,
#     image_path=im_path,
#     output_path=inc_map_out_path,
# )

# ################### UNDISTORT IMAGES ##################
# print("=== Undistorting images ===")
# undistort_out_path = os.path.join(DATASET_PATH, "dense")
# os.makedirs(undistort_out_path, exist_ok=True)

# pycolmap.undistort_images(
#     output_path=undistort_out_path,
#     input_path=os.path.join(inc_map_out_path, "0"),  # first reconstruction
#     image_path=im_path,
#     output_type="COLMAP",
# )


# # Check triangulation worked
# rec_path = "sparse/0"
# rec = pycolmap.Reconstruction(rec_path)
# print("Registered images:", len(rec.images))
# print("3D points:", len(rec.points3D))

# # To save point cloud 
# # rec = pycolmap.Reconstruction(rec_path)
# rec.export_PLY(os.path.join(rec_path, "points.ply"))
# print("Saved sparse point cloud to", os.path.join(rec_path, "points.ply"))


# ################### CPU LIMITATION ###################
# print("=== Skipping dense reconstruction (PatchMatch Stereo, Fusion, Meshing) because CUDA is required ===")

# ################### VISUALISATION #####################
# # Visualize sparse reconstruction
# sparse_ply = os.path.join(inc_map_out_path, "0", "points.ply")
# if os.path.exists(sparse_ply):
#     print("=== Loading and visualizing sparse point cloud ===")
#     pcd = o3d.io.read_point_cloud(sparse_ply)
#     o3d.visualization.draw_geometries([pcd])
# else:
#     print(f"No sparse point cloud found at {sparse_ply}. Sparse mapping might have failed.")



# ################### OPTIONAL: visualize first image with OpenCV SIFT ################
# # This is just to see extracted keypoints for debugging/CPU check
# first_image_path = os.path.join(im_path, im_names[0])
# img = cv2.imread(first_image_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp = sift.detect(gray, None)
# img_kp = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.figure(figsize=(10,8))
# plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
# plt.title("First image SIFT keypoints (CPU)")
# plt.axis('off')
# plt.show()

