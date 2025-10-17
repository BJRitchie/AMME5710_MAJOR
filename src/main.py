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


################### PATCH MATCH STEREO ###################
print("=== Running Patch Match Stereo ===")
wkspace_path = dense_path

pycolmap.patch_match_stereo(
    workspace_path=wkspace_path,
    workspace_format="COLMAP",
    options=pycolmap.PatchMatchOptions(),
)

################### STEREO FUSION ###################
print("=== Running Stereo Fusion ===")
stereo_outpath = os.path.join(dense_path, "fused.ply")

pycolmap.stereo_fusion(
    output_path=stereo_outpath,
    workspace_path=wkspace_path,
    workspace_format="COLMAP",
    input_type="geometric",
)

################### POISSON MESHER ###################

print("=== Running Poisson Meshing ===")
poisson_outpath = os.path.join(dense_path, "meshed-poisson.ply")

pycolmap.poisson_meshing(
    input_path=stereo_outpath,
    output_path=poisson_outpath,
)

################### DELAUNAY MESHER ###################
print("=== Running Delaunay Meshing ===")
delaunay_outpath = os.path.join(dense_path, "meshed-delaunay.ply")

pycolmap.dense_delaunay_meshing(
    input_path=wkspace_path,
    output_path=delaunay_outpath,
)

################### VISUALISATION ###################
print("=== Loading and visualizing final point cloud ===")
pcd = o3d.io.read_point_cloud(stereo_outpath)
o3d.visualization.draw_geometries([pcd])














# """
# From website tutorial https://mashaan14.github.io/YouTube-channel/nerf/2025_01_25_sfm.html

# @inproceedings{schoenberger2016sfm,
#  author     = {Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
#  title      = {Structure-from-Motion Revisited},
#  booktitle  = {Conference on Computer Vision and Pattern Recognition (CVPR)},
#  year       = {2016},
# }
# @inproceedings{wang2024vggsfm,
#  title      = {VGGSfM: Visual Geometry Grounded Deep Structure From Motion},
#  author     = {Wang, Jianyuan and Karaev, Nikita and Rupprecht, Christian and Novotny, David},
#  booktitle  = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#  pages      = {21686--21697},
#  year       = {2024}
# }

# """

# import pycolmap
# import numpy as np
# import cv2
# import os 
# import open3d as o3d

# ######################################################
# ################### INITIALISATION ###################
# ######################################################

# # The function InitializeReconstruction inside 
#     # colmap/src/colmap/controllers/incremental_pipeline.cc initializes 
#     # the reconstruction by calling FindInitialImagePair in 
#     # colmap/src/colmap/sfm/incremental_mapper.cc


# # EstimateCalibratedTwoViewGeometry: estimates two-view geometry from calibrated image pair.
#     # Extract corresponding points
#     # Estimate epipolar models
#     # Estimate planar or panoramic model
#     # Determine inlier ratios of different models

# # EstimateTwoViewGeometryPose: estimates relative pose for two-view geometry.
#     # Try to recover relative pose for calibrated and uncalibrated
#     # configurations. In the uncalibrated case, this most likely leads to a
#     # ill-defined reconstruction, but sometimes it succeeds anyways after e.g.
#     # subsequent bundle-adjustment etc.

# # PoseFromEssentialMatrix: recovers the most probable pose from the given essential matrix.
#     # Decompose an essential matrix into the possible rotations and translations.
#     # 
#     # The first pose is assumed to be P = [I | 0] and the set of four other
#     # possible second poses are defined as: {[R1 | t], [R2 | t],
#     #                                        [R1 | -t], [R2 | -t]}
#     # 
#     # @param E          3x3 essential matrix.
#     # @param R1         First possible 3x3 rotation matrix.
#     # @param R2         Second possible 3x3 rotation matrix.
#     # @param t          3x1 possible translation vector (also -t possible).

# # CheckCheirality
#     # Perform cheirality constraint test, i.e., determine which of the triangulated
#     # correspondences lie in front of both cameras.
#     # 
#     # @param cam2_from_cam1  Relative camera transformation.
#     # @param points1         First set of corresponding points.
#     # @param points2         Second set of corresponding points.
#     # @param points3D        Points that lie in front of both cameras.


# ##########################################################
# ################### IMAGE REGISTRATION ###################
# ##########################################################

# # New images can be registered to the current model by solving the 
#     # Perspective-n-Point (PnP) problem using feature correspondences to 
#     # triangulated points in already registered images (2D-3D correspondences)

# # The PnP problem involves estimating the pose Pc and, for uncalibrated cameras, 
#     # its intrinsic parameters. The set {curly P} is thus extended by the pose Pc of the newly registered 
#     # image (Schönberger and Frahm, 2016).


# # FindNextImages: sort images in a way that prioritize images with a sufficient number 
#     # of visible points.
# # RegisterNextImage
#     # search for 2D-3D correspondences
#     # estimate camera parameters
#     # pose refinement
#     # extend tracks to the newly registered image



# #####################################################
# ################### TRIANGULATION ###################
# #####################################################

# # A newly registered image must observe existing scene points. In addition, it may 
#     # also increase scene coverage by extending the set of points 𝒳through triangulation. 
#     # A new scene point Xk can be triangulated and added to 𝒳 as soon as at least one 
#     # more image, also covering the new scene part but from a different viewpoint, 
#     # is registered (Schönberger and Frahm, 2016).



# #########################################################
# ################### BUNDLE ADJUSTMENT ###################
# #########################################################

# # Without further refinement, SfM usually drifts quickly to a non-recoverable state. Bundle 
#     # adjustment is the joint non-linear refinement of camera parameters Pc and point parameters 
#     # Xk that minimizes the reprojection error: 
#     # E = \sum_j \rho_j ( || \pi (P_c, X_k) - x_j ||^2_2 ) 
#         # \pi: a function that projects scene points into image space
#         # \rho_j: the Cauchy function as the robust loss function to potentially down-weight outliers 
        
# # IterativeLocalRefinement: iteratively calls AdjustLocalBundle

# # AdjustLocalBundle
#     # Adjust locally connected images and points of a reference image. In
#     # addition, refine the provided 3D points. Only images connected to the
#     # reference image are optimized. If the provided 3D points are not locally
#     # connected to the reference image, their observing images are set as
#     # constant in the adjustment.

# # IterativeGlobalRefinement: iteratively calls AdjustGlobalBundle
# # AdjustGlobalBundle: Global bundle adjustment using Ceres Solver, which is usually used to solve Non-linear Least Squares problems.



# """
# Example datasets: https://demuc.de/colmap/datasets/ 
# Colmap command line interface tutorial: 

# # The project folder must contain a folder "images" with all the images.
# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense

# $ colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/dense \
#     --output_type COLMAP \
#     --max_image_size 2000

# $ colmap patch_match_stereo \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true

# $ colmap stereo_fusion \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path $DATASET_PATH/dense/fused.ply

# $ colmap poisson_mesher \
#     --input_path $DATASET_PATH/dense/fused.ply \
#     --output_path $DATASET_PATH/dense/meshed-poisson.ply

# $ colmap delaunay_mesher \
#     --input_path $DATASET_PATH/dense \
#     --output_path $DATASET_PATH/dense/meshed-delaunay.ply

# """




# # Read in dataset ... 
# DATASET_PATH = ""
# db_path = os.path.join( DATASET_PATH, "dataset.db" )
# im_path = os.path.join( DATASET_PATH, "images" )
# im_names = 

# cam_mode = 
# cam_model = 
# reader_ops = 
# sift_ops = 
# device = 

# # Extract features 
# pycolmap.extract_features(
#     database_path = db_path, 
#     image_path =  im_path,          #: str,
#     # image_names = im_names,         #: list[str] = [],
#     # camera_mode = cam_mode,         #: pycolmap.CameraMode = CameraMode.AUTO,
#     # camera_model = cam_model,       #: str = 'SIMPLE_RADIAL',
#     # reader_options = reader_ops,    #: pycolmap.ImageReaderOptions = ImageReaderOptions(),
#     # sift_options = sift_ops,        #: pycolmap.SiftExtractionOptions = SiftExtractionOptions(),
#     # device = device                 #: pycolmap.Device = Device.auto,
# ) 

# # Match features 
# pycolmap.match_exhaustive(
#     database_path = db_path,            # str,
#     # sift_options = sift_ops,            # pycolmap.SiftMatchingOptions = SiftMatchingOptions(),
#     # matching_options = matching_ops,    # pycolmap.ExhaustiveMatchingOptions = ExhaustiveMatchingOptions(),
#     # verification_options = verif_ops,   # pycolmap.TwoViewGeometryOptions = TwoViewGeometryOptions(),
#     # device = device                     # pycolmap.Device = Device.auto,
# ) 

# # Mapper 
# inc_map_out_path = os.path.join( DATASET_PATH, "incremental_mapping_out" ) 
# pycolmap.incremental_mapping(
#     database_path = db_path,                        # str,
#     image_path = im_path,                           # str,
#     output_path = inc_map_out_path,                         # str,
#     # options = inc_pipeline_ops,                     # pycolmap.IncrementalPipelineOptions = IncrementalPipelineOptions(),
#     # input_path = in_path,                           # str = '',
#     # initial_image_pair_callback = im_pair_callback, # Callable[[], None] = None,
#     # next_image_callback = next_im_callback          # Callable[[], None] = None,
# ) 

# # Undistort images 
# undistort_out_path = os.path.join( DATASET_PATH, "undistort_out" ) 
# pycolmap.undistort_images(
#     output_path = undistort_out_path,               # str, 
#     input_path = input_path,                        # str,
#     image_path = im_path,                           # str,
#     # image_names = im_names,                         # list[str] = [],
#     # output_type = out_type,                         # str = 'COLMAP',
#     # copy_policy = copy_pol,                         # pycolmap.CopyType = CopyType.copy,
#     # num_patch_match_src_images = num_patch_match,   # int = 20,
#     # undistort_options = undist_ops,                 # pycolmap.UndistortCameraOptions = UndistortCameraOptions(),
# ) 

# # Patch match stereo 
# pycolmap.patch_match_stereo(
#     workspace_path = wkspace_path,              # str,
#     # workspace_format = ,                      # str = 'COLMAP',
#     # pmvs_option_name = ,                      # str = 'option-all',
#     # options = ,                               # pycolmap.PatchMatchOptions = PatchMatchOptions(),
#     # config_path = ,                           # str = '',
# ) 

# # Stereo Fusion
# pycolmap.stereo_fusion(
#     output_path = stereo_outpath,  
#     workspace_path = wkspace_path, 
#     # workspace_format = ,  # str = 'COLMAP',
#     # pmvs_option_name = ,  # str = 'option-all',
#     # input_type = ,        # str = 'geometric',
#     # options = ,           # pycolmap.StereoFusionOptions = StereoFusionOptions()
# ) 

# # Poisson Mesher 
# pycolmap.poisson_meshing(
#     input_path = poisson_inpath, 
#     output_path = poisson_outpath, 
#     # options: pycolmap.PoissonMeshingOptions = PoissonMeshingOptions(),
# ) 

# # Delaunay Mesher 
# pycolmap.dense_delaunay_meshing(
#     input_path = delaunay_inpath, 
#     output_path = delaunay_outpath, 
#     # options: pycolmap.DelaunayMeshingOptions = DelaunayMeshingOptions(),
# )


