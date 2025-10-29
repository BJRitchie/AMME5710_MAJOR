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