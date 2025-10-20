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
    # def plot_pointcloud(self): 
        
    #     if os.path.exists(self._points_ply):
    #         print("=== Loading and visualizing sparse point cloud ===")
    #         pcd = o3d.io.read_point_cloud(self._points_ply)
    #         o3d.visualization.draw_geometries([pcd])
    #     else:
    #         print(f"No sparse point cloud found at {self._points_ply}. Sparse mapping might have failed.")
        
    #     return 
    
    def plot_pointcloud(self, store_name="0", camera_scale=0.1): 
        
        if not os.path.exists(self._points_ply):
            print(f"No sparse point cloud found at {self._points_ply}. Sparse mapping might have failed.")
            return
        
        print("=== Loading and visualizing sparse point cloud with cameras ===")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(self._points_ply)
        
        # Load reconstruction to get camera poses
        rec_path = os.path.join(self._sparse_path, store_name)
        rec = pycolmap.Reconstruction(rec_path)
        
        # Create camera frustum visualizations
        geometries = [pcd]
        
        for image_id, image in rec.images.items():
            # Get camera pose using the cam_from_world transformation
            cam_from_world = image.cam_from_world()
            
            # Get rotation matrix and translation from Rigid3d
            R = cam_from_world.rotation.matrix()  # 3x3 rotation matrix (world to camera)
            tvec = cam_from_world.translation  # Translation vector
            
            # Get camera center in world coordinates
            t = image.projection_center().flatten()  # Camera center in world space
            
            # Create coordinate frame for camera
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
            
            # Transform coordinate frame to camera pose
            # Convert rotation matrix to 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R.T  # Transpose because we want world orientation
            T[:3, 3] = t
            
            camera_frame.transform(T)
            geometries.append(camera_frame)
            
            # Optionally create a camera frustum pyramid
            camera = rec.cameras[image.camera_id]
            
            # Get image dimensions
            width = camera.width
            height = camera.height
            
            # Create frustum lines (simplified pyramid)
            frustum_depth = camera_scale * 2
            
            # Calculate frustum corners in camera space
            # Get focal length from camera parameters
            params = camera.params
            if len(params) >= 2:
                fx = params[0]
                fy = params[1]
            else:
                fx = fy = params[0] if len(params) > 0 else width
            
            cx = width / 2
            cy = height / 2
            
            # Frustum corners at depth
            corners_cam = np.array([
                [0, 0, 0],  # Camera center
                [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
                [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
                [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
                [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
            ])
            
            # Transform to world space
            corners_world = (R.T @ corners_cam.T).T + t
            
            # Create line set for frustum
            lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
            colors = [[1, 0, 0] for _ in lines]  # Red color for cameras
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners_world)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            geometries.append(line_set)
        
        print(f"Visualizing {len(rec.images)} cameras and {len(pcd.points)} points")
        o3d.visualization.draw_geometries(geometries)
        
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




