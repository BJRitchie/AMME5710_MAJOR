import os
import numpy as np
import open3d as o3d
import pycolmap



# Unscaled but just matched indice points
def plot_pointcloud_matched(sparse_path, store_path="0", camera_scale=0.1, matched_indices=None):
    """
    Visualize a COLMAP sparse reconstruction (points + camera frustums), optionally only for matched poses.

    Args:
        sparse_path (str): Path to the parent sparse directory (e.g., "path/to/sparse").
        store_path (str): Subfolder name (e.g., "0") containing the reconstruction and points.ply.
        camera_scale (float): Scale factor for visualizing camera coordinate frames and frustums.
        matched_indices (list[int], optional): List of indices of images to plot. If None, plots all cameras.
    """
    import os
    import open3d as o3d
    import pycolmap
    import numpy as np

    print("=== Loading and visualizing sparse point cloud with cameras ===")

    # Construct paths
    store_name = os.path.join(sparse_path, store_path)
    file_path = os.path.join(store_name, "points.ply")

    # Load point cloud
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"points.ply not found at {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    # Load reconstruction (cameras + poses)
    rec = pycolmap.Reconstruction(store_name)

    # Sort images to make indexing consistent
    images_sorted = sorted(rec.images.values(), key=lambda x: x.name)

    # If matched_indices is not provided, plot all images
    if matched_indices is None:
        matched_indices = list(range(len(images_sorted)))

    # Prepare visualization geometries
    geometries = [pcd]

    for idx in matched_indices:
        image = images_sorted[idx]
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()  # world → camera
        t = image.projection_center().flatten()  # camera center in world space

        # --- Coordinate frame ---
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        T = np.eye(4)
        T[:3, :3] = R.T  # world orientation
        T[:3, 3] = t
        camera_frame.transform(T)
        geometries.append(camera_frame)

        # --- Camera frustum ---
        camera = rec.cameras[image.camera_id]
        width, height = camera.width, camera.height
        frustum_depth = camera_scale * 2

        params = camera.params
        if len(params) >= 2:
            fx, fy = params[0], params[1]
        else:
            fx = fy = params[0] if len(params) > 0 else width

        cx = width / 2
        cy = height / 2

        corners_cam = np.array([
            [0, 0, 0],  # camera origin
            [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
            [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
        ])

        corners_world = (R.T @ corners_cam.T).T + t

        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [[1, 0, 0] for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)

    print(f"Visualizing {len(matched_indices)} cameras and {len(pcd.points)} points")
    o3d.visualization.draw_geometries(geometries)




# import os
# import numpy as np
# import open3d as o3d
# import pycolmap


def plot_pointcloud_orig(sparse_path, store_path="0", camera_scale=0.1):
    """
    Visualize a COLMAP sparse reconstruction (points + camera frustums).

    Args:
        sparse_path (str): Path to the parent sparse directory (e.g., "path/to/sparse").
        store_path (str): Subfolder name (e.g., "0") containing the reconstruction and points.ply.
        camera_scale (float): Scale factor for visualizing camera coordinate frames and frustums.
    """
    print("=== Loading and visualizing sparse point cloud with cameras ===")

    # Construct paths
    store_name = os.path.join(sparse_path, store_path)
    file_path = os.path.join(store_name, "points.ply")

    # Load point cloud
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"points.ply not found at {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    # Load reconstruction (cameras + poses)
    rec = pycolmap.Reconstruction(store_name)

    # Prepare visualization geometries
    geometries = [pcd]

    for image_id, image in rec.images.items():
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()  # world → camera
        t = image.projection_center().flatten()  # camera center in world space

        # --- Coordinate frame ---
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        T = np.eye(4)
        T[:3, :3] = R.T  # world orientation
        T[:3, 3] = t
        camera_frame.transform(T)
        geometries.append(camera_frame)

        # --- Camera frustum ---
        camera = rec.cameras[image.camera_id]
        width, height = camera.width, camera.height
        frustum_depth = camera_scale * 2

        params = camera.params
        if len(params) >= 2:
            fx, fy = params[0], params[1]
        else:
            fx = fy = params[0] if len(params) > 0 else width

        cx = width / 2
        cy = height / 2

        corners_cam = np.array([
            [0, 0, 0],  # camera origin
            [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
            [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
        ])

        corners_world = (R.T @ corners_cam.T).T + t

        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [[1, 0, 0] for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)

    print(f"Visualizing {len(rec.images)} cameras and {len(pcd.points)} points")
    o3d.visualization.draw_geometries(geometries)




def plot_pointcloud_scaled(sparse_path, sfm_rotations, sfm_translations_scaled, store_path="0", camera_scale=0.1):
    """
    Visualize a COLMAP sparse reconstruction with scaled SfM camera translations.

    Args:
        sparse_path (str): Path to the parent sparse directory (e.g., "path/to/sparse").
        sfm_rotations (list of 3x3 np.ndarray): Rotation matrices from SfM.
        sfm_translations_scaled (list of 3x1 np.ndarray): Translations already scaled.
        store_path (str): Subfolder name (e.g., "0") containing points.ply.
        camera_scale (float): Scale factor for visualizing camera coordinate frames and frustums.
    """
    import os
    import open3d as o3d
    import numpy as np

    print("=== Loading and visualizing sparse point cloud with scaled SfM cameras ===")

    # Load point cloud
    store_name = os.path.join(sparse_path, store_path)
    file_path = os.path.join(store_name, "points.ply")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"points.ply not found at {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    geometries = [pcd]

    for R, t in zip(sfm_rotations, sfm_translations_scaled):
        t = t.flatten()
        # Coordinate frame
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        camera_frame.rotate(R, center=(0, 0, 0))
        camera_frame.translate(t)
        geometries.append(camera_frame)

        # Frustum (dummy intrinsics)
        frustum_depth = camera_scale * 2
        width, height = 640, 480
        fx = fy = 500.0
        cx, cy = width / 2, height / 2

        corners_cam = np.array([
            [0, 0, 0],
            [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
            [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth]
        ])
        corners_world = (R @ corners_cam.T).T + t
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [[1,0,0] for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    print(f"Visualizing {len(sfm_translations_scaled)} scaled SfM cameras and {len(pcd.points)} points")
    o3d.visualization.draw_geometries(geometries)




import open3d as o3d
import numpy as np
import os

def plot_checkerboard_camera_poses(sparse_path, T_sat_to_cam_list, camera_scale=0.1):
    """
    Visualize a COLMAP sparse point cloud with checkerboard-derived camera poses in satellite frame.

    Args:
        sparse_path (str): Path to sparse reconstruction folder (e.g., "sparse/0").
        T_sat_to_cam_list (list of 4x4 np.ndarray): Each camera pose as homogeneous matrix in satellite frame.
        camera_scale (float): Scale for coordinate frames and camera frustums.
    """
    # Load point cloud
    file_path = os.path.join(sparse_path, "points.ply")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"points.ply not found at {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    geometries = [pcd]

    for T in T_sat_to_cam_list:
        R = T[:3, :3]
        t = T[:3, 3]

        # Camera coordinate frame
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        cam_frame.rotate(R, center=(0, 0, 0))
        cam_frame.translate(t)
        geometries.append(cam_frame)

        # Camera frustum (dummy image size and intrinsics)
        frustum_depth = camera_scale * 2
        width, height = 640, 480
        fx, fy = 500.0, 500.0
        cx, cy = width / 2, height / 2

        corners_cam = np.array([
            [0, 0, 0],
            [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
            [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
            [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
        ])
        corners_world = (R @ corners_cam.T).T + t
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [[1,0,0] for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    print(f"Visualizing {len(T_sat_to_cam_list)} checkerboard cameras and {len(pcd.points)} points")
    o3d.visualization.draw_geometries(geometries, window_name="Checkerboard Camera Poses")









def plot_sfm_vs_checkerboard(sparse_path, 
                             sfm_rotations, 
                             sfm_translations, 
                             checker_rotations, 
                             checker_translations, 
                             T_cb_to_sat, 
                             matched_indices_cb=None, 
                             camera_scale=0.1):
    """
    Plot COLMAP SfM point cloud with both SfM camera poses and checkerboard-derived poses.

    Args:
        sparse_path (str): Path to sparse reconstruction folder (e.g., "sparse/0").
        sfm_rotations (list of 3x3 np.ndarray): SfM rotation matrices.
        sfm_translations (list of 3x1 np.ndarray): SfM translations (before scaling).
        checker_rotations (list of 3x3 np.ndarray): Checkerboard rotation matrices.
        checker_translations (list of 3x1 np.ndarray): Checkerboard translations.
        T_cb_to_sat (4x4 np.ndarray): Transform from checkerboard to satellite frame.
        matched_indices_cb (list[int], optional): Indices of checkerboard poses corresponding to SfM images.
        camera_scale (float): Size of camera frames and frustums.
    """

    # -----------------------------
    # 1. Transform checkerboard poses to satellite frame
    # -----------------------------
    T_sat_to_cam_list = []
    for R_cb, t_cb in zip(checker_rotations, checker_translations):
        T_cb_to_cam = np.eye(4)
        T_cb_to_cam[:3, :3] = R_cb
        T_cb_to_cam[:3, 3] = t_cb.flatten()
        T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
        T_sat_to_cam_list.append(T_sat_to_cam)

    # -----------------------------
    # 2. Compute scale from matched translations
    # -----------------------------
    if matched_indices_cb is not None:
        sfm_t_array = np.hstack([sfm_translations[i] for i in matched_indices_cb])
        cb_t_array = np.hstack([T_sat_to_cam_list[i][:3,3].reshape(3,1) for i in matched_indices_cb])

        sfm_t_centered = sfm_t_array - sfm_t_array[:, [0]]
        cb_t_centered = cb_t_array - cb_t_array[:, [0]]

        scale = np.sum(cb_t_centered * sfm_t_centered) / np.sum(sfm_t_centered**2)
    else:
        scale = 1.0

    sfm_translations_scaled = [scale * t for t in sfm_translations]

    # -----------------------------
    # 3. Load point cloud
    # -----------------------------
    store_name = sparse_path
    file_path = os.path.join(store_name, "points.ply")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"points.ply not found at {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    geometries = [pcd]

    # -----------------------------
    # 4. Plot SfM cameras (red)
    # -----------------------------
    for R, t in zip(sfm_rotations, sfm_translations_scaled):
        t = t.flatten()
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        cam_frame.rotate(R, center=(0,0,0))
        cam_frame.translate(t)
        geometries.append(cam_frame)

        # Frustum
        frustum_depth = camera_scale * 2
        width, height = 640, 480
        fx, fy = 500.0, 500.0
        cx, cy = width/2, height/2

        corners_cam = np.array([
            [0,0,0],
            [(0-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
            [(width-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
            [(width-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth],
            [(0-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth]
        ])
        corners_world = (R @ corners_cam.T).T + t
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [[1,0,0] for _ in lines]  # red

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners_world)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(ls)

    # -----------------------------
    # 5. Plot checkerboard cameras (green)
    # -----------------------------
    for T in T_sat_to_cam_list:
        R = T[:3, :3]
        t = T[:3, 3]

        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        cam_frame.rotate(R, center=(0,0,0))
        cam_frame.translate(t)
        geometries.append(cam_frame)

        frustum_depth = camera_scale * 2
        width, height = 640, 480
        fx, fy = 500.0, 500.0
        cx, cy = width/2, height/2

        corners_cam = np.array([
            [0,0,0],
            [(0-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
            [(width-cx)*frustum_depth/fx, (0-cy)*frustum_depth/fy, frustum_depth],
            [(width-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth],
            [(0-cx)*frustum_depth/fx, (height-cy)*frustum_depth/fy, frustum_depth]
        ])
        corners_world = (R @ corners_cam.T).T + t
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [[0,1,0] for _ in lines]  # green

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners_world)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(ls)

    print(f"Visualizing {len(sfm_translations_scaled)} SfM cameras and {len(T_sat_to_cam_list)} checkerboard cameras")
    o3d.visualization.draw_geometries(geometries, window_name="SfM vs Checkerboard Camera Poses")































# def plot_pointcloud_checkerboard(sparse_path, checker_rotations, checker_translations, 
#                                  matched_indices_cb, camera_scale=0.1):
#     """
#     Visualize a COLMAP sparse reconstruction with camera poses derived from checkerboard calibration.

#     Args:
#         sparse_path (str): Path to sparse reconstruction folder (e.g., "sparse/0")
#         checker_rotations (list of np.ndarray): 3x3 rotation matrices (from checkerboard)
#         checker_translations (list of np.ndarray): 3x1 translations (from checkerboard)
#         matched_indices_cb (list of int): Indices of checkerboard poses that match SfM images
#         camera_scale (float): Scale for camera frames and frustums
#     """
#     store_name = sparse_path
#     file_path = os.path.join(store_name, "points.ply")

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"points.ply not found at {file_path}")
#     pcd = o3d.io.read_point_cloud(file_path)

#     geometries = [pcd]

#     # Transform checkerboard poses to satellite frame (example transform)
#     R_cb_to_sat = np.array([
#         [ 0.96917272, 0.0, 0.24638229],
#         [ 0.0,        1.0, 0.0      ],
#         [-0.24638229, 0.0, 0.96917272]
#     ])
#     t_cb_to_sat = np.array([[27.41], [0.0], [-35.25]]) / 1000.0
#     T_cb_to_sat = np.eye(4)
#     T_cb_to_sat[:3, :3] = R_cb_to_sat
#     T_cb_to_sat[:3, 3] = t_cb_to_sat.flatten()

#     for idx in matched_indices_cb:
#         R_cb = checker_rotations[idx]
#         t_cb = checker_translations[idx].flatten()

#         # Camera pose in satellite frame
#         T_cb_to_cam = np.eye(4)
#         T_cb_to_cam[:3, :3] = R_cb
#         T_cb_to_cam[:3, 3] = t_cb
#         T_sat_to_cam = T_cb_to_cam @ np.linalg.inv(T_cb_to_sat)
#         R = T_sat_to_cam[:3, :3]
#         t = T_sat_to_cam[:3, 3]

#         # --- Coordinate frame ---
#         cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         cam_frame.rotate(R, center=(0, 0, 0))
#         cam_frame.translate(t)
#         geometries.append(cam_frame)

#         # --- Camera frustum ---
#         frustum_depth = camera_scale * 2
#         width, height = 640, 480  # dummy image size for frustum
#         fx, fy = 500.0, 500.0      # dummy focal lengths
#         cx, cy = width / 2, height / 2

#         corners_cam = np.array([
#             [0, 0, 0],
#             [(0 - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (0 - cy) * frustum_depth / fy, frustum_depth],
#             [(width - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#             [(0 - cx) * frustum_depth / fx, (height - cy) * frustum_depth / fy, frustum_depth],
#         ])
#         corners_world = (R @ corners_cam.T).T + t
#         lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
#         colors = [[1,0,0] for _ in lines]

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)
#         geometries.append(line_set)

#     print(f"Visualizing {len(matched_indices_cb)} checkerboard cameras and {len(pcd.points)} points")
#     o3d.visualization.draw_geometries(geometries, window_name="Checkerboard Camera Poses")














# def plot_pointcloud(sparse_path, store_path="0", camera_scale=0.1, matched_indices=None):
#     """
#     Visualize a COLMAP sparse reconstruction (points + camera frustums),
#     optionally only for matched poses. Frustum size is scaled consistently.

#     Args:
#         sparse_path (str): Path to the parent sparse directory.
#         store_path (str): Subfolder containing reconstruction and points.ply.
#         camera_scale (float): Visual scale for axes and frustums.
#         matched_indices (list[int], optional): Indices of images to plot. If None, plot all.
#     """
#     print("=== Loading and visualizing sparse point cloud with cameras ===")

#     # Load point cloud
#     store_name = os.path.join(sparse_path, store_path)
#     file_path = os.path.join(store_name, "points.ply")
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"points.ply not found at {file_path}")
#     pcd = o3d.io.read_point_cloud(file_path)

#     # Load reconstruction
#     rec = pycolmap.Reconstruction(store_name)
#     images_sorted = sorted(rec.images.values(), key=lambda x: x.name)

#     # If no matched_indices provided, plot all images
#     if matched_indices is None:
#         matched_indices = list(range(len(images_sorted)))

#     geometries = [pcd]

#     frustum_depth = camera_scale * 2  # Depth of frustum from camera

#     for idx in matched_indices:
#         image = images_sorted[idx]
#         cam_from_world = image.cam_from_world()
#         R = cam_from_world.rotation.matrix()
#         t = image.projection_center().flatten()

#         # --- Coordinate frame ---
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         T = np.eye(4)
#         T[:3, :3] = R.T
#         T[:3, 3] = t
#         camera_frame.transform(T)
#         geometries.append(camera_frame)

#         # --- Camera frustum (fixed small visual size) ---
#         # Using simple [-0.5,0.5] offsets scaled by camera_scale
#         corners_cam = np.array([
#             [0, 0, 0],  # camera center
#             [-0.5, -0.5, 1.0],
#             [ 0.5, -0.5, 1.0],
#             [ 0.5,  0.5, 1.0],
#             [-0.5,  0.5, 1.0],
#         ]) * camera_scale * 2  # scale for visual size
#         corners_world = (R.T @ corners_cam.T).T + t

#         lines = [[0, 1], [0, 2], [0, 3], [0, 4],
#                  [1, 2], [2, 3], [3, 4], [4, 1]]
#         colors = [[1, 0, 0] for _ in lines]

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)
#         geometries.append(line_set)

#     print(f"Visualizing {len(matched_indices)} cameras and {len(pcd.points)} points")
#     o3d.visualization.draw_geometries(geometries)



# def plot_pointcloud_checkerboard(point_cloud, T_sat_to_cam_list, frustum_scale=0.05):
#     """
#     Visualize a point cloud with camera frustums from checkerboard-derived poses.

#     Args:
#         point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
#         T_sat_to_cam_list (list of 4x4 np.array): List of checkerboard-to-satellite camera transforms.
#         frustum_scale (float): Visual scale for camera frustums (smaller is better for compact visualization).
#     """
#     import open3d as o3d
#     import numpy as np

#     geometries = [point_cloud]

#     for T in T_sat_to_cam_list:
#         R = T[:3, :3]
#         t = T[:3, 3].flatten()

#         # --- Camera frustum (smaller, no axes) ---
#         corners_cam = np.array([
#             [0, 0, 0],       # camera center
#             [-0.5, -0.5, 1.0],
#             [0.5, -0.5, 1.0],
#             [0.5, 0.5, 1.0],
#             [-0.5, 0.5, 1.0],
#         ]) * frustum_scale * 2

#         corners_world = (R @ corners_cam.T).T + t

#         lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
#         colors = [[0, 1, 0] for _ in lines]  # Green for checkerboard

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)
#         geometries.append(line_set)

#     print(f"Visualizing {len(T_sat_to_cam_list)} checkerboard cameras and {len(point_cloud.points)} points")
#     o3d.visualization.draw_geometries(geometries)


















# def plot_pointcloud_checkerboard(point_cloud, T_sat_to_cam_list, camera_scale=0.1):
#     """
#     Visualize a point cloud with camera frustums from checkerboard-derived poses.

#     Args:
#         point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
#         T_sat_to_cam_list (list of 4x4 np.array): List of checkerboard-to-satellite camera transforms.
#         camera_scale (float): Visual scale for axes and frustums.
#     """
#     import open3d as o3d
#     import numpy as np

#     geometries = [point_cloud]

#     for T in T_sat_to_cam_list:
#         R = T[:3, :3]
#         t = T[:3, 3].flatten()

#         # --- Coordinate frame ---
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
#         T_frame = np.eye(4)
#         T_frame[:3, :3] = R
#         T_frame[:3, 3] = t
#         camera_frame.transform(T_frame)
#         geometries.append(camera_frame)

#         # --- Camera frustum (small) ---
#         corners_cam = np.array([
#             [0, 0, 0],
#             [-0.5, -0.5, 1.0],
#             [0.5, -0.5, 1.0],
#             [0.5, 0.5, 1.0],
#             [-0.5, 0.5, 1.0],
#         ]) * camera_scale * 2
#         corners_world = (R @ corners_cam.T).T + t

#         lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
#         colors = [[0, 1, 0] for _ in lines]  # Green for checkerboard

#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(corners_world)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)
#         geometries.append(line_set)

#     print(f"Visualizing {len(T_sat_to_cam_list)} checkerboard cameras and {len(point_cloud.points)} points")
#     o3d.visualization.draw_geometries(geometries)






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






