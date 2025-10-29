import pycolmap
 
import sfm_pipeline_lib as pipeline 
import gen_synthetic_pcd_lib as gen

## GENERATE SFM CLASS ##
# Convert the video into images 
store_path="images/batmo"

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


## GENERATE REF SYNTHETIC POINT CLOUD ##
# TODO Think these are also computing a target point cloud and gt, but then doing that again the function below - what if GT doesn't match??
# TODO Delete synthetic test data function and add in from here? Or make it consistent 
# Best working cubesat
ref_pcd = gen.generate_ppf_friendly_cubesat(num_points=20000, noise_std=0.001)


# Other shapes: (mostly didn't work well) - currently these autogenerate their own target perturbed point clouds, but ignore it is overwritten by function below (haven't bothered to fix in this functions since we aren't using them)
# ref_pcd, _, _ = gen.generate_test_pcds() # Basic box, also worked well 
# ref_pcd, _, _ = gen.generate_complex_test_pcds() # Some alignment, was ok
# ref_pcd, _, _ = gen.generate_sfm_sat_pcd(sfm_pipeline, num_points=20000) # Our own CAD model point cloud 
# ref_pcd, _, _ = gen.generate_ppf_friendly_satellite()
# ref_pcd, _, _ = gen.generate_complex_satellite_pcd()
# ref_pcd, _, _ = gen.generate_synthetic_satellite()

## GENERATE TEST SYNTHETIC POINT CLOUD ##
test_data = sfm_pipeline.generate_synthetic_test_data_from_pcd(ref_pcd, rotation_degrees=[5,5,5], translation=[0.1,0.2,0.05]) #  , noise_level = 0.001)

# Verify PPF
success = sfm_pipeline.verify_ppf_with_provided_pcds(
    test_data["ref_pcd"],
    test_data["synthetic_pcd"],
    test_data["transform_gt"]
)


# I don't really know what the "success" measure means
# But alignment metric errors are good

