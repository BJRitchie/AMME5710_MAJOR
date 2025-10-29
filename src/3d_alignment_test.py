import open3d as o3d
import numpy as np

def align_and_compare_pcds(sfm_path, cad_path, voxel_size=0.002):
    """
    Compare SfM-reconstructed and CAD-derived point clouds:
    - Estimate scale factor automatically
    - Align with ICP
    - Visualize before & after alignment

    Args:
        sfm_path (str): Path to SfM point cloud (.ply)
        cad_path (str): Path to CAD-derived point cloud (.ply)
        voxel_size (float): Voxel downsample size (meters)
    """

    # -----------------------------
    # 1. Load point clouds
    # -----------------------------
    print("[INFO] Loading point clouds...")
    pcd_sfm = o3d.io.read_point_cloud(sfm_path)
    pcd_cad = o3d.io.read_point_cloud(cad_path)

    print(f"  SfM points: {len(pcd_sfm.points)}")
    print(f"  CAD points: {len(pcd_cad.points)}")

    # -----------------------------
    # 2. Downsample (lightly)
    # -----------------------------
    # print("[INFO] Downsampling for efficiency...")
    # pcd_sfm_down = pcd_sfm.voxel_down_sample(voxel_size)
    # pcd_cad_down = pcd_cad.voxel_down_sample(voxel_size)
    # print(f"  SfM points (down): {len(pcd_sfm_down.points)}")
    # print(f"  CAD points (down): {len(pcd_cad_down.points)}")
    pcd_sfm_down = pcd_sfm
    pcd_cad_down = pcd_cad

    # -----------------------------
    # 3. Center both clouds
    # -----------------------------
    sfm_center = np.mean(np.asarray(pcd_sfm_down.points), axis=0)
    cad_center = np.mean(np.asarray(pcd_cad_down.points), axis=0)
    pcd_sfm_down.translate(-sfm_center)
    pcd_cad_down.translate(-cad_center)

    # -----------------------------
    # 4. Estimate scale factor (RMS-based)
    # -----------------------------
    sfm_extent = np.linalg.norm(np.max(np.asarray(pcd_sfm_down.points), axis=0) -
                                np.min(np.asarray(pcd_sfm_down.points), axis=0))
    cad_extent = np.linalg.norm(np.max(np.asarray(pcd_cad_down.points), axis=0) -
                                np.min(np.asarray(pcd_cad_down.points), axis=0))
    scale_factor = cad_extent / sfm_extent
    print(f"[INFO] Estimated scale factor: {scale_factor:.6f}")

    pcd_sfm_down.scale(scale_factor, center=np.zeros(3))

    # -----------------------------
    # 5. Rough PCA-based orientation alignment
    # -----------------------------
    print("[INFO] Performing PCA rough alignment...")
    def pca_align(pcd):
        pts = np.asarray(pcd.points)
        cov = np.cov(pts.T)
        _, vecs = np.linalg.eigh(cov)
        return vecs

    R_sfm = pca_align(pcd_sfm_down)
    R_cad = pca_align(pcd_cad_down)
    R_init = R_cad @ R_sfm.T
    pcd_sfm_down.rotate(R_init)

    # -----------------------------
    # 6. Visualize before ICP
    # -----------------------------
    print("[INFO] Visualizing BEFORE ICP alignment...")
    pcd_sfm_down.paint_uniform_color([1, 0, 0])  # SfM = red
    pcd_cad_down.paint_uniform_color([0, 1, 0])  # CAD = green

    o3d.visualization.draw_geometries(
        [pcd_sfm_down, pcd_cad_down],
        window_name="Before ICP Alignment (Red: SfM, Green: CAD)",
        width=1024,
        height=768
    )

    # -----------------------------
    # 7. Fine alignment using ICP
    # -----------------------------
    print("[INFO] Running ICP fine alignment...")
    threshold = voxel_size * 15
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd_sfm_down, pcd_cad_down, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    pcd_sfm_aligned = pcd_sfm_down.transform(reg_icp.transformation)

    # -----------------------------
    # 8. Compute alignment error
    # -----------------------------
    distances = pcd_sfm_aligned.compute_point_cloud_distance(pcd_cad_down)
    rmse = np.sqrt(np.mean(np.square(distances)))

    print("\n[RESULTS]")
    print("ICP Transformation Matrix:\n", reg_icp.transformation)
    print(f"Alignment RMSE: {rmse:.6f} m")

    # -----------------------------
    # 9. Visualize after ICP
    # -----------------------------
    print("[INFO] Visualizing AFTER ICP alignment...")
    pcd_sfm_aligned.paint_uniform_color([1, 0, 0])  # Red = aligned SfM
    pcd_cad_down.paint_uniform_color([0, 1, 0])     # Green = CAD
    o3d.visualization.draw_geometries(
        [pcd_sfm_aligned, pcd_cad_down],
        window_name="After ICP Alignment (Red: SfM, Green: CAD)",
        width=1024,
        height=768
    )

    return reg_icp.transformation, scale_factor, rmse


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sfm_path = "sparse/0/points.ply"      # SfM reconstruction file
    cad_path = "reference.ply"     # CAD model file

    T_icp, scale_factor, rmse = align_and_compare_pcds(sfm_path, cad_path)
    print(f"\nFinal scale factor: {scale_factor:.6f}")
    print(f"Final RMSE: {rmse:.6f} m")
