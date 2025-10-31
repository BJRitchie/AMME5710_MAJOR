import numpy as np
import open3d as o3d
import copy
import os 



def generate_test_pcds():
    # Create a non-symmetric shape with good geometry
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
    mesh.compute_vertex_normals()
    ref_pcd = mesh.sample_points_poisson_disk(5000)

    # Define a known transform (small rotation + translation)
    angle = np.deg2rad(15)
    R = ref_pcd.get_rotation_matrix_from_xyz((angle, angle / 2, angle / 3))
    t = np.array([0.2, 0.1, 0.05])
    transform_gt = np.eye(4)
    transform_gt[:3, :3] = R
    transform_gt[:3, 3] = t

    # Apply transform to get target point cloud
    target_pcd = copy.deepcopy(ref_pcd).transform(transform_gt)

    # Add tiny bit of noise to make features unique
    target_pts = np.asarray(target_pcd.points)
    target_pts += np.random.normal(scale=0.001, size=target_pts.shape)
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)


    # Visualize the reference and target clouds
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])   # green (reference)

    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1]) # red (transformed)

    # Combine them for visualization
    o3d.visualization.draw_geometries(
        [ref_vis, target_vis],
        window_name="Reference (green) vs Transformed (red)",
        width=900,
        height=700,
        point_show_normal=False
    )

    return ref_pcd, target_pcd, transform_gt



def generate_synthetic_satellite():
    """
    Generate a satellite-like point cloud with planar panels
    """
    # Create box panels to simulate satellite body
    body = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
    panel1 = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.02, depth=0.6)
    panel1.translate((0.0, 0.3, 0.0))
    panel2 = copy.deepcopy(panel1).translate((0.8, 0.3, 0.0))
    
    mesh = body + panel1 + panel2
    mesh.compute_vertex_normals()
    
    # Sample points densely
    ref_pcd = mesh.sample_points_poisson_disk(15000)
    
    # Apply small rotation + translation
    angle = np.deg2rad([5.0, 5.0, 5.0])
    R = ref_pcd.get_rotation_matrix_from_xyz(angle)
    t = np.array([0.1, 0.2, 0.05])
    T_gt = np.eye(4)
    T_gt[:3,:3] = R
    T_gt[:3,3] = t
    
    target_pcd = copy.deepcopy(ref_pcd).transform(T_gt)
    
    # Add small noise
    pts = np.asarray(target_pcd.points)
    pts += np.random.normal(scale=0.001, size=pts.shape)
    target_pcd.points = o3d.utility.Vector3dVector(pts)
    
    return ref_pcd, target_pcd, T_gt


def generate_complex_test_pcds():
    """
    Generate a more complex point cloud shape to test alignment algorithms.
    Combines cylinder, cone, and small sphere in an asymmetric layout.
    """
    # Create components
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=0.6)
    cylinder.translate((-0.3, 0, 0))
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.15, height=0.5)
    cone.translate((0.3, 0, 0))
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.translate((0.0, 0.3, 0.1))
    
    # Combine meshes
    mesh = cylinder + cone + sphere
    mesh.compute_vertex_normals()
    
    # Sample points densely
    ref_pcd = mesh.sample_points_poisson_disk(8000)
    
    # Apply known transformation
    angle = np.deg2rad([20, 10, 15])
    R = ref_pcd.get_rotation_matrix_from_xyz(angle)
    t = np.array([0.15, 0.2, 0.05])
    transform_gt = np.eye(4)
    transform_gt[:3, :3] = R
    transform_gt[:3, 3] = t
    
    target_pcd = copy.deepcopy(ref_pcd).transform(transform_gt)
    
    # Add small noise
    target_pts = np.asarray(target_pcd.points)
    target_pts += np.random.normal(scale=0.001, size=target_pts.shape)
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)
    
    # Visualize
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1])
    
    o3d.visualization.draw_geometries(
        [ref_vis, target_vis],
        window_name="Reference (green) vs Transformed (red)",
        width=900,
        height=700,
        point_show_normal=False
    )
    
    return ref_pcd, target_pcd, transform_gt





def generate_complex_satellite_pcd(num_points=20000, noise_std=0.001):
    """
    Generate a more complex satellite-like point cloud for testing PPF + ICP.
    Includes body, solar panels, antennas, and asymmetry.
    """
    # Central body
    body = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
    body.translate((-0.5, -0.3, -0.2))

    # Solar panels
    panel1 = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.02, depth=0.8)
    panel1.translate((0.0, 0.3, -0.4))
    
    panel2 = copy.deepcopy(panel1).translate((0.8, 0.0, 0.0))

    # Antenna / protrusions
    antenna = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.3)
    antenna.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=(0,0,0))
    antenna.translate((0.1, 0.6, 0.1))

    # Combine meshes
    mesh = body + panel1 + panel2 + antenna
    mesh.compute_vertex_normals()

    # Sample points
    ref_pcd = mesh.sample_points_poisson_disk(num_points)

    # Apply known rotation + translation
    angle = np.deg2rad([10, 5, 15])
    R = ref_pcd.get_rotation_matrix_from_xyz(angle)
    t = np.array([0.15, 0.2, 0.05])
    T_gt = np.eye(4)
    T_gt[:3,:3] = R
    T_gt[:3,3] = t

    target_pcd = copy.deepcopy(ref_pcd).transform(T_gt)

    # Add small noise
    pts = np.asarray(target_pcd.points)
    pts += np.random.normal(scale=noise_std, size=pts.shape)
    target_pcd.points = o3d.utility.Vector3dVector(pts)

    # Visualize
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1])

    o3d.visualization.draw_geometries(
        [ref_vis, target_vis],
        window_name="Complex Satellite: Reference (green) vs Transformed (red)",
        width=900,
        height=700
    )

    return ref_pcd, target_pcd, T_gt





def generate_ppf_friendly_satellite(num_points=20000, noise_std=0.001):
    """
    Generate a synthetic satellite-like point cloud that works well with PPF matching.
    """
    # --- Create main body ---
    body = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.4)
    body.translate((-0.5, -0.3, -0.2))  # center at origin

    # --- Add thick panels for PPF-friendly features ---
    panel1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.4, depth=0.6)  # left
    panel1.translate((-0.6, -0.2, -0.3))

    panel2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.4, depth=0.6)  # right
    panel2.translate((0.6, -0.2, -0.3))

    panel3 = o3d.geometry.TriangleMesh.create_box(width=0.6, height=0.02, depth=0.4)  # top
    panel3.translate((-0.3, 0.3, -0.2))

    panel4 = o3d.geometry.TriangleMesh.create_box(width=0.6, height=0.02, depth=0.4)  # bottom
    panel4.translate((-0.3, -0.32, -0.2))

    # Combine meshes
    mesh = body + panel1 + panel2 + panel3 + panel4
    mesh.compute_vertex_normals()

    # --- Sample points ---
    ref_pcd = mesh.sample_points_poisson_disk(num_points)
    
    # --- Apply known transformation ---
    angles = np.deg2rad([5.0, 5.0, 5.0])  # small rotation
    R = ref_pcd.get_rotation_matrix_from_xyz(angles)
    t = np.array([0.1, 0.2, 0.05])
    T_gt = np.eye(4)
    T_gt[:3, :3] = R
    T_gt[:3, 3] = t

    target_pcd = copy.deepcopy(ref_pcd).transform(T_gt)

    # --- Add small Gaussian noise ---
    target_pts = np.asarray(target_pcd.points)
    target_pts += np.random.normal(scale=noise_std, size=target_pts.shape)
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    # --- Optional visualization ---
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])
    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1])
    
    o3d.visualization.draw_geometries([ref_vis, target_vis],
                                      window_name="PPF-friendly Satellite",
                                      width=900, height=700)
    
    return ref_pcd, target_pcd, T_gt





def generate_sfm_sat_pcd(sfm_pipeline, num_points):

    # To get satellite
    sfm_pipeline.make_reference_ply()

    # Load reference model
    ref_path = os.path.join("sat_model", "reference.ply")
    if not os.path.exists(ref_path):
        print(f"Reference model not found at {ref_path}")
        print("Run make_reference_ply() first")
        
    ref_mesh = o3d.io.read_triangle_mesh(ref_path)

    # Sample points from reference model
    ref_pcd = ref_mesh.sample_points_uniformly(number_of_points=num_points)
    print(f"Sampled {len(ref_pcd.points)} points from reference model")

    return ref_pcd








def generate_ppf_friendly_cubesat(num_points=20000, noise_std=0.001):
    """
    Generate a synthetic CubeSat-like point cloud suitable for PPF matching.
    """
    # --- Create CubeSat main body (cube) ---
    body = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.3, depth=0.3)
    body.translate((-0.15, -0.15, -0.15))  # center at origin

    # --- Add two solar panels extending from opposite sides ---
    panel_width = 0.02
    panel_height = 0.2
    panel_depth = 0.4

    panel1 = o3d.geometry.TriangleMesh.create_box(width=panel_width, height=panel_height, depth=panel_depth)
    panel1.translate((-0.16, -0.1, -0.05))  # left

    panel2 = o3d.geometry.TriangleMesh.create_box(width=panel_width, height=panel_height, depth=panel_depth)
    panel2.translate((0.14, -0.1, -0.05))   # right

    # --- Add an asymmetric antenna stub for uniqueness ---
    antenna = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.1)
    antenna.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)))
    antenna.translate((0.0, 0.15, 0.0))

    # --- Combine meshes ---
    mesh = body + panel1 + panel2 + antenna
    mesh.compute_vertex_normals()

    # --- Sample points ---
    ref_pcd = mesh.sample_points_poisson_disk(num_points)

    # --- Optional visualization ---
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])  # green

    o3d.visualization.draw_geometries([ref_vis],
                                      window_name="PPF-friendly CubeSat",
                                      width=900, height=700)
    
    return ref_pcd # , target_pcd, T_gt



def generate_test_pcds_sat():
    # Create a box of dimensions 3cm x 2cm x 1cm (converted to meters)
    width = 0.195 # 0.03   # 3 cm 19.5cm
    height = 0.105 # 0.02  # 2 cm 10.5cm
    depth = 0.082 # 0.01   # 1 cm 8.2cm
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    mesh.compute_vertex_normals()

    # Sample points from the mesh
    ref_pcd = mesh.sample_points_poisson_disk(3000)

    # Define a known small transform (rotation + translation)
    angle = np.deg2rad(15)
    R = ref_pcd.get_rotation_matrix_from_xyz((angle, angle / 2, angle / 3))
    t = np.array([0.005, 0.003, 0.002])  # small translation (5 mm, 3 mm, 2 mm)
    transform_gt = np.eye(4)
    transform_gt[:3, :3] = R
    transform_gt[:3, 3] = t

    # Apply transform to get target point cloud
    target_pcd = copy.deepcopy(ref_pcd).transform(transform_gt)

    # Add small noise to make the point sets non-identical
    target_pts = np.asarray(target_pcd.points)
    target_pts += np.random.normal(scale=0.0001, size=target_pts.shape)  # 0.1 mm noise
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    # Color and visualize both
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])   # green (reference)

    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1]) # red (transformed)

    o3d.visualization.draw_geometries(
        [ref_vis, target_vis],
        window_name="3 cm x 2 cm x 1 cm Box: Reference (green) vs Transformed (red)",
        width=900,
        height=700,
        point_show_normal=False
    )

    return ref_pcd # , target_pcd, transform_gt


# def generate_data_from_sfm(ref_pcd, sfm_points_path="sparse/0/points.ply"):
#     """
#     Load SfM point cloud, scale it to roughly match the reference cloud size,
#     translate both clouds to the reference center, and visualize.

#     Plots:
#         1. Point clouds only
#         2. Point clouds with oriented bounding boxes (OBBs)

#     Args:
#         ref_pcd: Open3D PointCloud object (reference CAD or test cloud)
#         sfm_points_path: path to SfM-generated point cloud (.ply)

#     Returns:
#         dict: containing reference PCD, scaled & centered SfM PCD, and dummy transform_gt
#     """
#     import open3d as o3d
#     import numpy as np
#     import os

#     # --- Load SfM cloud ---
#     if not os.path.exists(sfm_points_path):
#         raise FileNotFoundError(f"SfM point cloud not found at: {sfm_points_path}")
#     sfm_pcd = o3d.io.read_point_cloud(sfm_points_path)
#     if len(sfm_pcd.points) == 0:
#         raise ValueError(f"Loaded SfM point cloud from '{sfm_points_path}' is empty.")

#     # --- Compute OBBs ---
#     ref_obb = ref_pcd.get_oriented_bounding_box()
#     sfm_obb = sfm_pcd.get_oriented_bounding_box()

#     # --- Scale SfM cloud to match reference size ---
#     scale_factor = np.linalg.norm(ref_obb.extent) / np.linalg.norm(sfm_obb.extent)
#     sfm_center = sfm_obb.center
#     sfm_pcd.translate(-sfm_center)
#     sfm_pcd.scale(scale_factor, center=(0, 0, 0))

#     # --- Translate both clouds to reference center ---
#     ref_center = ref_obb.center
#     sfm_pcd.translate(ref_center)

#     print(f"[INFO] Scaled SfM cloud by factor {scale_factor:.4f} and moved to reference center.")

#     # --- Plot 1: point clouds only ---
#     ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
#     o3d.visualization.draw_geometries(
#         [ref_vis, sfm_pcd],
#         window_name="Reference (green) vs SfM (original colors)",
#         width=900, height=700
#     )

#     # --- Plot 2: point clouds with OBBs ---
#     ref_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(ref_obb)
#     ref_obb_ls.paint_uniform_color([0, 1, 0])  # green wireframe

#     # Recompute OBB after scaling & translation
#     sfm_obb = sfm_pcd.get_oriented_bounding_box()
#     sfm_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(sfm_obb)
#     sfm_obb_ls.paint_uniform_color([1, 0, 0])

#     sfm_obb_ls.paint_uniform_color([1, 0, 0])  # red wireframe

#     o3d.visualization.draw_geometries(
#         [ref_vis, ref_obb_ls, sfm_pcd, sfm_obb_ls],
#         window_name="Reference (green) vs SfM (original colors) + OBBs",
#         width=900, height=700
#     )

#     # --- Dummy ground-truth transform ---
#     transform_gt = np.eye(4)

#     return {
#         "ref_pcd": ref_pcd,
#         "synthetic_pcd": sfm_pcd,
#         "transform_gt": transform_gt
#     }


# def generate_data_from_sfm(ref_pcd, sfm_points_path="sparse/0/points.ply"):
#     """
#     Load SfM point cloud, scale it to roughly match the reference cloud size,
#     translate both clouds to the reference center, and visualize.

#     Plots:
#         1. Point clouds only
#         2. Point clouds with oriented bounding boxes (OBBs)

#     Args:
#         ref_pcd: Open3D PointCloud object (reference CAD or test cloud)
#         sfm_points_path: path to SfM-generated point cloud (.ply)

#     Returns:
#         dict: containing reference PCD, scaled & centered SfM PCD, and dummy transform_gt
#     """
#     import open3d as o3d
#     import numpy as np
#     import os

#     # --- Load SfM cloud ---
#     if not os.path.exists(sfm_points_path):
#         raise FileNotFoundError(f"SfM point cloud not found at: {sfm_points_path}")
#     sfm_pcd = o3d.io.read_point_cloud(sfm_points_path)
#     if len(sfm_pcd.points) == 0:
#         raise ValueError(f"Loaded SfM point cloud from '{sfm_points_path}' is empty.")

#     # --- Compute OBBs ---
#     ref_obb = ref_pcd.get_oriented_bounding_box()
#     sfm_obb = sfm_pcd.get_oriented_bounding_box()

#     print(f"[INFO] Reference OBB dimensions: {ref_obb.extent}")
#     print(f"[INFO] Original SfM OBB dimensions: {sfm_obb.extent}")

#     # --- Scale SfM cloud to match reference size ---
#     scale_factor = np.linalg.norm(ref_obb.extent) / np.linalg.norm(sfm_obb.extent)
#     sfm_center = sfm_obb.center
#     sfm_pcd.translate(-sfm_center)
#     sfm_pcd.scale(scale_factor, center=(0, 0, 0))

#     # --- Translate both clouds to reference center ---
#     ref_center = ref_obb.center
#     sfm_pcd.translate(ref_center)

#     # Recompute OBB after scaling & translation
#     sfm_obb = sfm_pcd.get_oriented_bounding_box()
#     print(f"[INFO] Scaled & centered SfM OBB dimensions: {sfm_obb.extent}")

#     # --- Plot 1: point clouds only ---
#     ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
#     o3d.visualization.draw_geometries(
#         [ref_vis, sfm_pcd],
#         window_name="Reference (green) vs SfM (original colors)",
#         width=900, height=700
#     )

#     # --- Plot 2: point clouds with OBBs ---
#     ref_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(ref_obb)
#     ref_obb_ls.paint_uniform_color([0, 1, 0])  # green wireframe

#     sfm_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(sfm_obb)
#     sfm_obb_ls.paint_uniform_color([1, 0, 0])  # red wireframe

#     o3d.visualization.draw_geometries(
#         [ref_vis, ref_obb_ls, sfm_pcd, sfm_obb_ls],
#         window_name="Reference (green) vs SfM (original colors) + OBBs",
#         width=900, height=700
#     )

#     # --- Dummy ground-truth transform ---
#     transform_gt = np.eye(4)

#     return {
#         "ref_pcd": ref_pcd,
#         "synthetic_pcd": sfm_pcd,
#         "transform_gt": transform_gt
#     }


# Scales each axis independently (could stretch/shrink in 3 axes)
def generate_data_from_sfm(ref_pcd, sfm_points_path="sparse/0/points.ply"):
    """
    Load SfM point cloud, scale it to exactly match the reference cloud's OBB dimensions,
    translate both clouds to the reference center, and visualize.

    Plots:
        1. Point clouds only
        2. Point clouds with oriented bounding boxes (OBBs)

    Args:
        ref_pcd: Open3D PointCloud object (reference CAD or test cloud)
        sfm_points_path: path to SfM-generated point cloud (.ply)

    Returns:
        dict: containing reference PCD, scaled & centered SfM PCD, and dummy transform_gt
    """
    import open3d as o3d
    import numpy as np
    import os

    # --- Load SfM cloud ---
    if not os.path.exists(sfm_points_path):
        raise FileNotFoundError(f"SfM point cloud not found at: {sfm_points_path}")
    sfm_pcd = o3d.io.read_point_cloud(sfm_points_path)
    if len(sfm_pcd.points) == 0:
        raise ValueError(f"Loaded SfM point cloud from '{sfm_points_path}' is empty.")

    # --- Compute OBBs ---
    ref_obb = ref_pcd.get_oriented_bounding_box()
    sfm_obb = sfm_pcd.get_oriented_bounding_box()

    print(f"[INFO] Reference OBB dimensions: {ref_obb.extent}")
    print(f"[INFO] Original SfM OBB dimensions: {sfm_obb.extent}")

    # --- Translate SfM cloud to origin ---
    sfm_center = sfm_obb.center
    sfm_pcd.translate(-sfm_center)

    # --- Scale SfM cloud per axis to match reference OBB dimensions ---
    scale_factors = ref_obb.extent / sfm_obb.extent
    sfm_points = np.asarray(sfm_pcd.points)
    sfm_points *= scale_factors  # non-uniform scaling per axis
    sfm_pcd.points = o3d.utility.Vector3dVector(sfm_points)

    # --- Translate SfM cloud to reference center ---
    ref_center = ref_obb.center
    sfm_pcd.translate(ref_center)

    # --- Recompute OBB after scaling & translation ---
    sfm_obb = sfm_pcd.get_oriented_bounding_box()
    print(f"[INFO] Scaled & centered SfM OBB dimensions: {sfm_obb.extent}")

    # --- Plot 1: point clouds only ---
    ref_vis = ref_pcd.paint_uniform_color([0.1, 0.8, 0.1])  # green
    o3d.visualization.draw_geometries(
        [ref_vis, sfm_pcd],
        window_name="Reference (green) vs SfM (original colors)",
        width=900, height=700
    )

    # --- Plot 2: point clouds with OBBs ---
    ref_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(ref_obb)
    ref_obb_ls.paint_uniform_color([0, 1, 0])  # green wireframe

    sfm_obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(sfm_obb)
    sfm_obb_ls.paint_uniform_color([1, 0, 0])  # red wireframe

    o3d.visualization.draw_geometries(
        [ref_vis, ref_obb_ls, sfm_pcd, sfm_obb_ls],
        window_name="Reference (green) vs SfM (original colors) + OBBs",
        width=900, height=700
    )

    # --- Dummy ground-truth transform ---
    transform_gt = np.eye(4)

    return {
        "ref_pcd": ref_pcd,
        "synthetic_pcd": sfm_pcd,
        "transform_gt": transform_gt
    }





def generate_test_pcds_plane():
    """
    Generate test data where the reference is a 3x2x1 cm box,
    and the synthetic (target) is a rotated & translated plane.
    """

    # --- Reference: 3cm x 2cm x 1cm box (in meters) ---
    width = 0.03   # 3 cm
    height = 0.02  # 2 cm
    depth = 0.01   # 1 cm
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    mesh_box.compute_vertex_normals()
    ref_pcd = mesh_box.sample_points_poisson_disk(3000)

    # --- Target: flat plane (3cm x 2cm) ---
    mesh_plane = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=1e-4)
    mesh_plane.compute_vertex_normals()
    plane_pcd = mesh_plane.sample_points_poisson_disk(3000)

    # --- Apply a small known rotation + translation to the plane ---
    angle = np.deg2rad(15)
    R = ref_pcd.get_rotation_matrix_from_xyz((angle, angle / 2, angle / 3))
    t = np.array([0.005, 0.003, 0.002])  # translation: 5mm, 3mm, 2mm
    transform_gt = np.eye(4)
    transform_gt[:3, :3] = R
    transform_gt[:3, 3] = t

    target_pcd = copy.deepcopy(plane_pcd).transform(transform_gt)

    # --- Add tiny noise for realism ---
    target_pts = np.asarray(target_pcd.points)
    target_pts += np.random.normal(scale=0.0001, size=target_pts.shape)  # 0.1 mm noise
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    # --- Visualization (optional) ---
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])   # green = reference box

    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1]) # red = rotated plane

    o3d.visualization.draw_geometries(
        [ref_vis, target_vis],
        window_name="Reference Box (green) vs Rotated Plane (red)",
        width=900,
        height=700,
        point_show_normal=False
    )

    # --- Package into dictionary ---
    test_data = {
        "ref_pcd": ref_pcd,
        "synthetic_pcd": target_pcd,
        "transform_gt": transform_gt
    }

    return test_data





def generate_test_pcds_bbs(num_points=5000):
    # Reference dimensions
    # ref_dims = np.array([0.20034846, 0.11800896, 0.09378805])
    ref_dims = np.array([0.19758827, 0.11123925, 0.08802385])

    # Original SfM dimensions (before scaling)
    # sfm_dims = np.array([0.1861358, 0.12237741, 0.11506107])
    sfm_dims = np.array([0.1975043, 0.10704513, 0.09446862])

    # --- Create reference mesh ---
    ref_mesh = o3d.geometry.TriangleMesh.create_box(
        width=ref_dims[0], height=ref_dims[1], depth=ref_dims[2]
    )
    ref_mesh.compute_vertex_normals()
    ref_pcd = ref_mesh.sample_points_poisson_disk(num_points)

    # --- Create target (SfM) mesh ---
    sfm_mesh = o3d.geometry.TriangleMesh.create_box(
        width=sfm_dims[0], height=sfm_dims[1], depth=sfm_dims[2]
    )
    sfm_mesh.compute_vertex_normals()
    target_pcd = sfm_mesh.sample_points_poisson_disk(num_points)

    # Apply a small random transform to make it misaligned
    angle = np.deg2rad(10)
    R = target_pcd.get_rotation_matrix_from_xyz((angle, angle / 2, angle / 3))
    t = np.array([0.05, -0.03, 0.02])
    transform_gt = np.eye(4)
    transform_gt[:3, :3] = R
    transform_gt[:3, 3] = t
    target_pcd.transform(transform_gt)

    # Add small noise for uniqueness
    pts = np.asarray(target_pcd.points)
    pts += np.random.normal(scale=0.001, size=pts.shape)
    target_pcd.points = o3d.utility.Vector3dVector(pts)

    # --- Visualize ---
    ref_vis = copy.deepcopy(ref_pcd)
    ref_vis.paint_uniform_color([0.1, 0.8, 0.1])   # green
    target_vis = copy.deepcopy(target_pcd)
    target_vis.paint_uniform_color([0.8, 0.1, 0.1]) # red

    o3d.visualization.draw_geometries(
        [ref_vis, target_vis],
        window_name="Reference (green) vs Target/SfM (red)",
        width=900, height=700
    )

    return {
        "ref_pcd": ref_pcd,
        "synthetic_pcd": target_pcd,
        "transform_gt": transform_gt
    }
