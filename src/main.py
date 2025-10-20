import pycolmap

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 

# Convert the video into images 
# vid_path = "images/batmo.mp4"
# vid_path = 'images/ben.mp4'
# store_path="images/ben"
vid_path = 'images/first_sat.mp4'
store_path="images/first_sat"
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
sfm_pipeline = pipeline.StrcFromMotion( 
    db_path, im_path, sparse_path, dense_path, 
    cam_mode    =pycolmap.CameraMode.AUTO, 
    cam_model   ="SIMPLE_RADIAL",  
    reader_ops  =pycolmap.ImageReaderOptions(), 
    sift_ops    =sift_ops, 
    device      =pycolmap.Device.cpu 
) 

sfm_pipeline.resize_ims( store_path, 1200, 5 )
sfm_pipeline.prep_pointcloud() 
sfm_pipeline.make_point_cloud()
sfm_pipeline.plot_pointcloud() 

