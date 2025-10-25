import pycolmap
import numpy as np
import cv2
import os
import shutil
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData

################### Plotting #####################    
def plot_pointcloud(store_path="0", camera_scale=0.1): 
        
    print("=== Loading and visualizing sparse point cloud with cameras ===")
    
    # Load point cloud 
    store_name = os.path.join( "sparse", store_path ) 
    file_path =  os.path.join( "sparse", store_path, "points.ply" )
    pcd = o3d.io.read_point_cloud( file_path ) 
    
    # Load reconstruction to get camera poses
    rec = pycolmap.Reconstruction( store_name )
    
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

plot_pointcloud()