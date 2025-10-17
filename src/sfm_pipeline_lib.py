import pycolmap
import numpy as np
import cv2
import os
import shutil
import open3d as o3d
import matplotlib.pyplot as plt

class StrcFromMotion: 
    def __init__(self, 
                 db_path:       str, 
                 im_path:       str, 
                 sparse_path:   str,
                 dense_path:    str,
                 cam_mode:      pycolmap.CameraMode, 
                 cam_model:     str, 
                 reader_ops:    pycolmap.ImageReaderOptions, 
                 sift_ops:      pycolmap.SiftExtractionOptions, 
                 device:        pycolmap.Device): 
        
        # Paths 
        self._database_path =db_path
        self._image_path    =im_path
        self._sparse_path   =sparse_path
        self._dense_path    =dense_path
        
        # Settings 
        self._camera_mode   =cam_mode
        self._camera_model  =cam_model
        self._reader_options=reader_ops
        self._sift_options  =sift_ops
        self._device        =device

        # Unassigned variables 
        self._image_names   =None 
        self._points_ply    =None
        
        # Clean up workspace 
        paths = [db_path, sparse_path, dense_path] 
        self._clean_up( paths )
        
        # Make fresh output directories 
        self._make_clean_dirs( [sparse_path, dense_path] )
        
        return 

    def resize_ims( self, im_path, max_size, interval=1 ): 
        
        # Get the image names 
        self._image_names = [f for f in os.listdir(im_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self._image_names = self._image_names[::interval] # Take only every Nth image 
        
        # Iterate over each file  
        for f in self._image_names:
            path = os.path.join(im_path, f)
            img = cv2.imread(path)
            
            if img is None:
                continue
            
            # Resize to fit common dimensions 
            h, w = img.shape[:2]
            scale = max_size / max(h, w) 
            if scale < 1.0:
                img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
                cv2.imwrite(path, img_resized)

    def prep_pointcloud( self ): 
        
        print("=== Extracting features ===")
        pycolmap.extract_features(
            database_path=  self._database_path,
            image_path=     self._image_path,
            image_names=    self._image_names,
            camera_mode=    self._camera_mode,
            camera_model=   self._camera_model,
            reader_options= self._reader_options,
            sift_options=   self._sift_options,
            device=         self._device 
        )
        
        print("=== Matching features (exhaustive) ===")
        pycolmap.match_exhaustive(
            database_path=          self._database_path,
            device=                 self._device,
            sift_options=           pycolmap.SiftMatchingOptions(),
            matching_options=       pycolmap.ExhaustiveMatchingOptions(),
            verification_options=   pycolmap.TwoViewGeometryOptions(),
        )

        print("=== Running incremental mapping ===")
        pycolmap.incremental_mapping(
            database_path=  self._database_path,
            image_path=     self._image_path,
            output_path=    self._sparse_path,
        )

        print("=== Undistorting images ===")
        pycolmap.undistort_images(
            output_path=    self._dense_path, 
            input_path=     os.path.join(self._sparse_path, "0"),  # first reconstruction
            image_path=     self._image_path,
            output_type=    "COLMAP",
        )

        return 

    def make_point_cloud( self, store_name="0" ): 
        
        # Read the reconstructed point cloud 
        rec_path = os.path.join(self._sparse_path, store_name)
        
        # Reconstruct the pointcloud 
        rec = pycolmap.Reconstruction(rec_path)
        
        print("Registered images:", len(rec.images))
        print("3D points:", len(rec.points3D))

        # Save the point cloud 
        self._points_ply = os.path.join(rec_path, "points.ply")
        rec.export_PLY(self._points_ply)
        print("Saved sparse point cloud to", self._points_ply)

        return 

    ################### Internal Funcs #####################
    def _clean_up(self, paths): 
        
        # Clean each of the paths 
        for path in paths:
            if os.path.exists(path):
                print(f"Removing old file/folder: {path}")
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)

        return 

    def _make_clean_dirs( self, paths ): 
        for path in paths: 
            os.makedirs(path, exist_ok=True)

        return 


    ################### Plotting #####################
    def plot_pointcloud(self): 
        
        if os.path.exists(self._points_ply):
            print("=== Loading and visualizing sparse point cloud ===")
            pcd = o3d.io.read_point_cloud(self._points_ply)
            o3d.visualization.draw_geometries([pcd])
        else:
            print(f"No sparse point cloud found at {self._points_ply}. Sparse mapping might have failed.")
        
        return 
    
    def plot_keypoints(self): 
        
        first_image_path = os.path.join(self._image_path, self._image_names[0])
        img = cv2.imread(first_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        img_kp = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title("First image SIFT keypoints (CPU)")
        plt.axis('off')
        plt.show()

        return 




