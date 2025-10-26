import pycolmap

# My packages 
import sfm_pipeline_lib as pipeline 
from file_reading_lib import gen_images_from_vid 

# Convert the video into images 
# vid_path = "images/batmo.mp4"
# vid_path = 'images/ben.mp4'
# store_path="images/ben"
vid_path = 'images/batmo.mp4'
store_path="images/batmo"
gen_images_from_vid( vid_path, store_path ) 

# Storage files 
im_path = store_path
db_path = "database.db"
sparse_path = "sparse"
dense_path = "dense"
sat_model_path = "sat_model"

# Settings 
sift_ops = pycolmap.SiftExtractionOptions()
sift_ops.use_gpu = False # CPU only 
sift_ops.first_octave = 0
sift_ops.num_octaves = 4

# Initialise the pipeline 
sfm_pipeline = pipeline.StrcFromMotion ( 
    db_path, im_path, sparse_path, dense_path, sat_model_path,
    cam_mode    =pycolmap.CameraMode.AUTO, 
    cam_model   ="SIMPLE_RADIAL",  
    reader_ops  =pycolmap.ImageReaderOptions(), 
    sift_ops    =sift_ops, 
    device      =pycolmap.Device.cpu 
) 

# sfm_pipeline.make_reference_ply()
# sfm_pipeline.plot_reference_model()

# sfm_pipeline.resize_ims( store_path, 1200, 10 )
# sfm_pipeline.prep_pointcloud() 
# sfm_pipeline.make_pointcloud()
# sfm_pipeline.clean_pointcloud() 
# sfm_pipeline.plot_pointcloud()

# Surface matching using PPF + ICP (run after point cloud generation)
# First, let's diagnose what went wrong with the previous attempt
print("=== DIAGNOSING PREVIOUS ALIGNMENT FAILURE ===")
sfm_pipeline.diagnose_alignment_failure(store_path="sat_model")

print("\n=== RUNNING IMPROVED SURFACE MATCHING ===")
# Run the improved surface matching with correct path
sfm_pipeline.surface_matching_ppf_icp(store_path="sat_model", voxel_size=0.05)

