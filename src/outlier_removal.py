import pycolmap
import matplotlib.pyplot as plt
import open3d as o3d

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 
import checkerboard_lib as checkerboard


import pickle

# Convert the video into images 
vid_path = 'images/bigger_checker.mp4'
store_path="images/bigger_checker"
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

sfm_pipeline.resize_ims( store_path, 1200, 2 )  # originally 10 but wasn't using images with checkerboard in it
sfm_pipeline.prep_pointcloud() 
sfm_pipeline.make_pointcloud()

with open("sfm_pipeline.pkl", "wb") as f:
    pickle.dump(sfm_pipeline, f)
print("Saved SfM pipeline to 'sfm_pipeline.pkl'")

# with open("sfm_pipeline.pkl", "rb") as f:
#     sfm_pipeline = pickle.load(f)

sfm_pipeline.plot_pointcloud() 

sfm_pipeline.clean_pointcloud(nb_points = 80, radius = 0.25) # TODO experiment with this a bit more - gets rid of far outliers but not close ones (just off of body)
# sfm_pipeline.stat_clean_pointcloud() # TODO experiment with this a bit more - gets rid of far outliers but not close ones (just off of body)

sfm_pipeline.plot_pointcloud() 


