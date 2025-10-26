import os
import numpy as np
import open3d as o3d
import pycolmap

def plot_pointcloud(sparse_path, store_path="0", camera_scale=0.1, matched_indices=None):
    """
    Visualize a COLMAP sparse reconstruction (points + camera frustums),
    optionally only for matched poses. Frustum size is scaled consistently.

    Args:
        sparse_path (str): Path to the parent sparse directory.
        store_path (str): Subfolder containing reconstruction and points.ply.
        camera_scale (float): Visual scale for axes and frustums.
        matched_indices (list[int], optional): Indices of images to plot. If None, plot all.
    """
    print("=== Loading and visualizing sparse point cloud with cameras ===")

    # Load point cloud
    store_name = os.path.join(sparse_path, store_path)
    file_path = os.path.join(store_name, "points.ply")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"points.ply not found at {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    # Load reconstruction
    rec = pycolmap.Reconstruction(store_name)
    images_sorted = sorted(rec.images.values(), key=lambda x: x.name)

    # If no matched_indices provided, plot all images
    if matched_indices is None:
        matched_indices = list(range(len(images_sorted)))

    geometries = [pcd]

    frustum_depth = camera_scale * 2  # Depth of frustum from camera

    for idx in matched_indices:
        image = images_sorted[idx]
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = image.projection_center().flatten()

        # --- Coordinate frame ---
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = t
        camera_frame.transform(T)
        geometries.append(camera_frame)

        # --- Camera frustum (fixed small visual size) ---
        # Using simple [-0.5,0.5] offsets scaled by camera_scale
        corners_cam = np.array([
            [0, 0, 0],  # camera center
            [-0.5, -0.5, 1.0],
            [ 0.5, -0.5, 1.0],
            [ 0.5,  0.5, 1.0],
            [-0.5,  0.5, 1.0],
        ]) * camera_scale * 2  # scale for visual size
        corners_world = (R.T @ corners_cam.T).T + t

        lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                 [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [[1, 0, 0] for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    print(f"Visualizing {len(matched_indices)} cameras and {len(pcd.points)} points")
    o3d.visualization.draw_geometries(geometries)



# def plot_pointcloud(
#     sparse_path,
#     store_path="0",
#     camera_scale=0.1,
#     matched_indices=None,
#     translations_scaled=None,  # NEW: provide the scaled translations
#     scale_pointcloud=1.0       # NEW: scale the point cloud by s
# ):
#     import os
#     import open3d as o3d
#     import pycolmap
#     import numpy as np

#     print("=== Loading and visualizing sparse point cloud with cameras ===")

#     # Load and scale point cloud
#     store_name = os.path.join(sparse_path, store_path)
#     file_path = os.path.join(store_name, "points.ply")
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"points.ply not found at {file_path}")

#     pcd = o3d.io.read_point_cloud(file_path)
#     points = np.asarray(pcd.points) * scale_pointcloud
#     colors = np.asarray(pcd.colors)
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     rec = pycolmap.Reconstruction(store_name)
#     images_sorted = sorted(rec.images.values(), key=lambda x: x.name)

#     if matched_indices is None:
#         matched_indices = list(range(len(images_sorted)))

#     geometries = [pcd]

#     for i, idx in enumerate(matched_indices):
#         img = images_sorted[idx]

#         # Use scaled translations if provided
#         if translations_scaled is not None:
#             t = translations_scaled[i].flatten()
#         else:
#             t = img.projection_center().flatten()

#         R = img.cam_from_world().rotation.matrix()
#         camera = rec.cameras[img.camera_id]
#         fx = camera.params[0] if len(camera.params) > 0 else 1
#         fy = camera.params[1] if len(camera.params) > 1 else fx
#         width, height = camera.width, camera.height

#         # Coordinate frame
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         T = np.eye(4)
#         T[:3, :3] = R.T
#         T[:3, 3] = t
#         camera_frame.transform(T)
#         geometries.append(camera_frame)

#         # Camera frustum
#         frustum_depth = camera_scale * 2
#         cx, cy = width / 2, height / 2
#         corners_cam = np.array([
#             [0, 0, 0],
#             [(0-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
#             [(width-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
#             [(width-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth],
#             [(0-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth],
#         ])
#         corners_world = (R.T @ corners_cam.T).T + t
#         lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
#         colors = [[1,0,0] for _ in lines]
#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)
#         geometries.append(line_set)

#     print(f"Visualizing {len(matched_indices)} cameras and {len(pcd.points)} points")
#     o3d.visualization.draw_geometries(geometries)








# Unscaled but just matched indice points
# def plot_pointcloud(sparse_path, store_path="0", camera_scale=0.1, matched_indices=None):
#     """
#     Visualize a COLMAP sparse reconstruction (points + camera frustums), optionally only for matched poses.

#     Args:
#         sparse_path (str): Path to the parent sparse directory (e.g., "path/to/sparse").
#         store_path (str): Subfolder name (e.g., "0") containing the reconstruction and points.ply.
#         camera_scale (float): Scale factor for visualizing camera coordinate frames and frustums.
#         matched_indices (list[int], optional): List of indices of images to plot. If None, plots all cameras.
#     """
#     import os
#     import open3d as o3d
#     import pycolmap
#     import numpy as np

#     print("=== Loading and visualizing sparse point cloud with cameras ===")

#     # Construct paths
#     store_name = os.path.join(sparse_path, store_path)
#     file_path = os.path.join(store_name, "points.ply")

#     # Load point cloud
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"points.ply not found at {file_path}")
#     pcd = o3d.io.read_point_cloud(file_path)

#     # Load reconstruction (cameras + poses)
#     rec = pycolmap.Reconstruction(store_name)

#     # Sort images to make indexing consistent
#     images_sorted = sorted(rec.images.values(), key=lambda x: x.name)

#     # If matched_indices is not provided, plot all images
#     if matched_indices is None:
#         matched_indices = list(range(len(images_sorted)))

#     # Prepare visualization geometries
#     geometries = [pcd]

#     for idx in matched_indices:
#         image = images_sorted[idx]
#         cam_from_world = image.cam_from_world()
#         R = cam_from_world.rotation.matrix()  # world → camera
#         t = image.projection_center().flatten()  # camera center in world space

#         # --- Coordinate frame ---
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         T = np.eye(4)
#         T[:3, :3] = R.T  # world orientation
#         T[:3, 3] = t
#         camera_frame.transform(T)
#         geometries.append(camera_frame)

#         # --- Camera frustum ---
#         camera = rec.cameras[image.camera_id]
#         width, height = camera.width, camera.height
#         frustum_depth = camera_scale * 2

#         params = camera.params
#         if len(params) >= 2:
#             fx, fy = params[0], params[1]
#         else:
#             fx = fy = params[0] if len(params) > 0 else width

#         cx = width / 2
#         cy = height / 2

#         corners_cam = np.array([
#             [0, 0, 0],  # camera origin
#             [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#             [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#         ])

#         corners_world = (R.T @ corners_cam.T).T + t

#         lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
#         colors = [[1, 0, 0] for _ in lines]

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)

#         geometries.append(line_set)

#     print(f"Visualizing {len(matched_indices)} cameras and {len(pcd.points)} points")
#     o3d.visualization.draw_geometries(geometries)




# import os
# import numpy as np
# import open3d as o3d
# import pycolmap


# def plot_pointcloud(sparse_path, store_path="0", camera_scale=0.1):
#     """
#     Visualize a COLMAP sparse reconstruction (points + camera frustums).

#     Args:
#         sparse_path (str): Path to the parent sparse directory (e.g., "path/to/sparse").
#         store_path (str): Subfolder name (e.g., "0") containing the reconstruction and points.ply.
#         camera_scale (float): Scale factor for visualizing camera coordinate frames and frustums.
#     """
#     print("=== Loading and visualizing sparse point cloud with cameras ===")

#     # Construct paths
#     store_name = os.path.join(sparse_path, store_path)
#     file_path = os.path.join(store_name, "points.ply")

#     # Load point cloud
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"points.ply not found at {file_path}")
#     pcd = o3d.io.read_point_cloud(file_path)

#     # Load reconstruction (cameras + poses)
#     rec = pycolmap.Reconstruction(store_name)

#     # Prepare visualization geometries
#     geometries = [pcd]

#     for image_id, image in rec.images.items():
#         cam_from_world = image.cam_from_world()
#         R = cam_from_world.rotation.matrix()  # world → camera
#         t = image.projection_center().flatten()  # camera center in world space

#         # --- Coordinate frame ---
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         T = np.eye(4)
#         T[:3, :3] = R.T  # world orientation
#         T[:3, 3] = t
#         camera_frame.transform(T)
#         geometries.append(camera_frame)

#         # --- Camera frustum ---
#         camera = rec.cameras[image.camera_id]
#         width, height = camera.width, camera.height
#         frustum_depth = camera_scale * 2

#         params = camera.params
#         if len(params) >= 2:
#             fx, fy = params[0], params[1]
#         else:
#             fx = fy = params[0] if len(params) > 0 else width

#         cx = width / 2
#         cy = height / 2

#         corners_cam = np.array([
#             [0, 0, 0],  # camera origin
#             [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#             [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#         ])

#         corners_world = (R.T @ corners_cam.T).T + t

#         lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
#         colors = [[1, 0, 0] for _ in lines]

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)

#         geometries.append(line_set)

#     print(f"Visualizing {len(rec.images)} cameras and {len(pcd.points)} points")
#     o3d.visualization.draw_geometries(geometries)



# plot_pointcloud("sparse", store_path="0", camera_scale=0.1)
