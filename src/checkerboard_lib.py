import numpy as np
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob 

class Checkerboard: 
    def __init__(self):
        
        self._ims = []  
        
        pass 
    
    def read_ims(self, im_path): 
        
        # Check it is a valid path 
        assert os.path.exists(im_path), f"{im_path} doesn't exist" 
        

        # Match common image file extensions
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.pgm')
        
        # Find all the files 
        files = [] 
        for ext in extensions:
            files.extend(glob.glob(os.path.join(im_path, ext)))
        
        # Iterate over each file 
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                
            if img is not None:  # Ensure it read correctly
                self._ims.append(img) 
        
        return  
    
    """Determines the intrinsic parameters of the camera using the checkerboard photos
    
    Args:
        grid_size (tuple): (cols, rows) representing the number of internal corners along the width and 
            height of the chessboard. It's important to note that this is the number of internal corners, 
            not the number of squares. For example, a standard 8x8 chessboard has 7x7 internal corners
            
        cell_size (int): real physical size of the cells in meters. 
            
        window_size (tuple): Half of the side length of the search window, e.g. (5,5) means an 11x11 search window is used
        
        criteria (tuple): Termination criteria for the iterative refinement process. This typically includes 
            a maximum number of iterations (cv2.TERM_CRITERIA_MAX_ITER) and/or a minimum accuracy threshold (cv2.TERM_CRITERIA_EPS). 
            The refinement stops when either condition is met.

    Returns:
        None: But it stores 
            self._K: camera instrinsic matrix. 
            self._d: camera distortion coefficients. 
            self._rvecs: rotation vectors to the checkerboard images. 
            self._tvecs: translation vector to the checkerboard images. 
        
    """
    def undistort_ims(self, 
                      grid_size, 
                      cell_size,
                      window_size=(11,11), 
                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    ): 
        assert (len(self._ims) > 0), "No images stored."
        
        # Record checkerboard params 
        self._grid_size= grid_size
        self._cell_size= cell_size

        checker_board_coords = [] 
        corr_3d_points = [] 

        # Iterate over all the stored images 
        for im in self._ims: 
            H, W = im.shape 
            
            # Detect and refine  
            ret, corners = cv2.findChessboardCorners(im, grid_size, None) 
            # ret, corners = cv2.findChessboardCorners(
            #     im, grid_size,
            #     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            # )
            if not ret:
                print(f"Checkerboard not found, skipping.")
                continue 
            corners2 = cv2.cornerSubPix(im, corners, window_size, (-1,-1), criteria)
            checker_board_coords.append(corners2) 

            # Build numpy array containing (x,y,z) coordinates of corners, relative to board itself
            pattern_points = np.zeros((np.prod(grid_size), 3), np.float32)
            pattern_points[:, :2] = np.indices(grid_size).T.reshape(-1, 2)
            pattern_points = cell_size * pattern_points
            corr_3d_points.append(pattern_points)

        # Calibrate the camera 
        output = cv2.calibrateCameraExtended(
            corr_3d_points, checker_board_coords, (W, H), None, None
        )
        # retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags = output
        
        # Store 
        self._K = output[1] # cameraMatrix 
        self._d = output[2] # distCoeffs, vector of distortion coefficients 
        self._rvecs = output[3] # Output vector of rotation vectors estimated for each pattern view 
        self._tvecs = output[4] # Output vector of translation vectors estimated for each pattern view 

        return 
    
    def plot_checkerboards(self):
        """
        Visualize the checkerboard poses estimated from calibration in 3D camera space using Open3D.
        """
        assert hasattr(self, "_grid_size") and hasattr(self, "_cell_size"), \
            "Grid size and cell size not defined."
        assert hasattr(self, "_rvecs") and hasattr(self, "_tvecs"), \
            "Run undistort_ims() first to compute rvecs and tvecs."

        # Prepare board corner points (same as during calibration)
        objp = np.zeros((np.prod(self._grid_size), 3), np.float32)
        objp[:, :2] = np.indices(self._grid_size).T.reshape(-1, 2)
        objp *= self._cell_size

        geometries = []

        # axis length for plotting
        axis_length = float(self._cell_size * self._grid_size[0] / 2)

        # Add camera origin coordinate frame
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 1.2)
        geometries.append(cam_frame)

        # Define a small palette for board point clouds
        colors = [
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            [0.8, 0.6, 0.2],
            [0.6, 0.2, 0.8],
            [0.2, 0.8, 0.6]
        ]

        for i, (rvec, tvec) in enumerate(zip(self._rvecs, self._tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            # Ensure tvec is shape (3,) float
            origin = np.asarray(tvec).reshape(3).astype(float)
            board_points = (R @ objp.T + origin.reshape(3, 1)).T.astype(float)

            # Point cloud for checkerboard corners
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(board_points)
            pcd.paint_uniform_color(colors[i % len(colors)])
            geometries.append(pcd)

            # Draw axes lines for the board (x=red, y=green, z=blue)
            axes = np.float64([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
            axes_transformed = (R @ axes.T + origin.reshape(3, 1)).T

            line_points = np.vstack([origin, axes_transformed])  # 4 points: origin, x, y, z
            lines = [[0, 1], [0, 2], [0, 3]]
            line_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(line_points)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(line_colors)
            geometries.append(ls)

            # Add a coordinate frame at the board origin oriented by R
            board_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length * 0.6)
            # rotate then translate
            board_frame.rotate(R, center=(0, 0, 0))
            board_frame.translate(origin)
            geometries.append(board_frame)

        # Visualize all geometries
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Checkerboard Poses (Camera Frame)",
            width=1024,
            height=768,
            mesh_show_back_face=True
        )

    def compute_scale(self, reconstructed_points, pattern_index=0):
        """
        Compute the metric scale factor for an unscaled 3D reconstruction,
        using the known physical size of the checkerboard.

        Args:
            reconstructed_points (np.ndarray): Nx3 array of reconstructed checkerboard corner points
                                            (in arbitrary units, same order as calibration detection).
            pattern_index (int): which checkerboard pose to reference (default=0)

        Returns:
            float: scale factor to convert reconstruction to metric units
        """
        assert hasattr(self, "_grid_size") and hasattr(self, "_cell_size"), \
            "Run undistort_ims() first to define grid and cell size."
        assert reconstructed_points.shape[1] == 3, \
            "reconstructed_points must be Nx3."

        # Build the true (known) checkerboard corner coordinates in meters
        objp = np.zeros((np.prod(self._grid_size), 3), np.float32)
        objp[:, :2] = np.indices(self._grid_size).T.reshape(-1, 2)
        objp *= self._cell_size

        # Compute distance between two adjacent corners along one axis
        real_dist = np.linalg.norm(objp[0] - objp[1])
        recon_dist = np.linalg.norm(reconstructed_points[0] - reconstructed_points[1])

        if recon_dist < 1e-9:
            raise ValueError("Reconstructed points are degenerate (zero distance detected).")

        scale_factor = real_dist / recon_dist
        print(f"[Scale] Real: {real_dist:.6f} m | Recon: {recon_dist:.6f} units | Scale factor: {scale_factor:.6f}")
        return scale_factor


    def apply_scale(self, points, scale_factor):
        """
        Apply the computed scale factor to a 3D reconstruction.

        Args:
            points (np.ndarray): Nx3 point cloud (unscaled)
            scale_factor (float): scale factor from compute_scale()

        Returns:
            np.ndarray: Nx3 scaled point cloud (metric)
        """
        assert points.shape[1] == 3, "points must be Nx3"
        return points * scale_factor



# if __name__ == "__main__":
#     import open3d as o3d
#     import numpy as np
#     from sklearn.cluster import KMeans
#     import matplotlib.pyplot as plt
#     import colorsys

#     # -----------------------------
#     # 1. Load point cloud
#     # -----------------------------
#     pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
#     points = np.asarray(pcd.points)
#     colors = np.asarray(pcd.colors)  # RGB in [0,1]
#     print(f"Loaded point cloud with {points.shape[0]} points")

#     # -----------------------------
#     # 2. Convert RGB to HSV
#     # -----------------------------
#     hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in colors])
    
#     # Plot HSV distribution
#     fig, axs = plt.subplots(1,3, figsize=(15,4))
#     axs[0].hist(hsv_colors[:,0], bins=50, color='r')
#     axs[0].set_title("Hue")
#     axs[1].hist(hsv_colors[:,1], bins=50, color='g')
#     axs[1].set_title("Saturation")
#     axs[2].hist(hsv_colors[:,2], bins=50, color='b')
#     axs[2].set_title("Value")
#     plt.show()

#     # -----------------------------
#     # 3. HSV thresholding
#     # -----------------------------
#     H_min, H_max = 0.1, 0.2
#     S_min, S_max = 0.5, 1.0
#     V_min, V_max = 0.2, 1.0

#     mask = ((hsv_colors[:,0] >= H_min) & (hsv_colors[:,0] <= H_max) &
#             (hsv_colors[:,1] >= S_min) & (hsv_colors[:,1] <= S_max) &
#             (hsv_colors[:,2] >= V_min) & (hsv_colors[:,2] <= V_max))
    
#     filtered_points = points[mask]
#     filtered_colors = colors[mask]
#     print(f"Filtered to {filtered_points.shape[0]} points based on HSV")

#     filtered_pcd = o3d.geometry.PointCloud()
#     filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
#     filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
#     o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Points by HSV")

#     # -----------------------------
#     # 4. Cluster filtered points (k=2)
#     # -----------------------------
#     kmeans = KMeans(n_clusters=2, random_state=42).fit(filtered_points)
#     labels = kmeans.labels_
    
#     centroids = kmeans.cluster_centers_
#     print(f"Centroids:\n{centroids}")
    
#     cluster_pcds = []
#     colors_palette = [[1,0,0], [0,1,0]]  # for visualization
#     for i in range(2):
#         cluster_points = filtered_points[labels==i]
#         pcd_cluster = o3d.geometry.PointCloud()
#         pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
#         pcd_cluster.paint_uniform_color(colors_palette[i])
#         cluster_pcds.append(pcd_cluster)

#     # -----------------------------
#     # 5. Draw line between centroids
#     # -----------------------------
#     line_points = centroids
#     lines = [[0,1]]
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(line_points),
#         lines=o3d.utility.Vector2iVector(lines)
#     )
#     line_set.paint_uniform_color([1,1,0])  # yellow line

#     o3d.visualization.draw_geometries(cluster_pcds + [line_set], window_name="Clusters with Centroid Line")

#     # -----------------------------
#     # 6. Compute distance and scale factor
#     # -----------------------------
#     known_distance = 0.05  # meters, for example
#     measured_distance = np.linalg.norm(centroids[0] - centroids[1])
#     scale_factor = known_distance / measured_distance
#     print(f"Measured distance: {measured_distance:.6f} → Scale factor: {scale_factor:.6f}")

#     # -----------------------------
#     # 7. Apply scale to entire point cloud
#     # -----------------------------
#     points_scaled = points * scale_factor
#     scaled_pcd = o3d.geometry.PointCloud()
#     scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
#     scaled_pcd.colors = o3d.utility.Vector3dVector(colors)
#     print("Applied metric scaling to full point cloud.")

#     # Visualize scaled point cloud
#     scaled_cluster_pcds = []
#     for i in range(2):
#         cluster_points_scaled = filtered_points[labels==i] * scale_factor
#         pcd_cluster_scaled = o3d.geometry.PointCloud()
#         pcd_cluster_scaled.points = o3d.utility.Vector3dVector(cluster_points_scaled)
#         pcd_cluster_scaled.paint_uniform_color(colors_palette[i])
#         scaled_cluster_pcds.append(pcd_cluster_scaled)

#     scaled_line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(centroids * scale_factor),
#         lines=o3d.utility.Vector2iVector([[0,1]])
#     )
#     scaled_line_set.paint_uniform_color([1,1,0])

#     o3d.visualization.draw_geometries([scaled_pcd] + scaled_cluster_pcds + [scaled_line_set],
#                                       window_name="Scaled Point Cloud with Centroid Line")



# K means clsutering with planar detection TODO or add in z height detection since plane should be consistent?
if __name__ == "__main__":
    import open3d as o3d
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # -----------------------------
    # 1. Calibrate camera using checkerboard images
    # -----------------------------
    cb = Checkerboard()
    cb.read_ims("images/sat_checkerboard_pgm")
    cb.undistort_ims(grid_size=(3, 3), cell_size=0.016)  
    cb.plot_checkerboards()

    # -----------------------------
    # 2. Load point cloud (object + checkerboard)
    # -----------------------------
    pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
    points = np.asarray(pcd.points)
    print(f"Loaded point cloud with {points.shape[0]} points")

    # -----------------------------
    # 3. Segment point cloud using K-means clustering (3 clusters)
    # -----------------------------
    kmeans = KMeans(n_clusters=3, random_state=42).fit(points)
    labels = kmeans.labels_

    clusters = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue

    for i in range(3):
        cluster_points = points[labels == i]
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
        pcd_cluster.paint_uniform_color(colors[i])
        clusters.append((cluster_points, pcd_cluster))
        print(f"Cluster {i}: {cluster_points.shape[0]} points")

    o3d.visualization.draw_geometries(
        [pcd for _, pcd in clusters],
        window_name="K-means Clusters (R/G/B)"
    )

    # -----------------------------
    # 4. Identify checkerboard candidate by planarity (keep original color)
    # -----------------------------
    best_plane_score = float("inf")
    checkerboard_points = None
    checkerboard_color = None  # store cluster color

    for i, (cluster_points, pcd_cluster) in enumerate(clusters):
        if len(cluster_points) < 10:
            continue  # ignore tiny clusters (noise)

        try:
            plane_model, inliers = pcd_cluster.segment_plane(
                distance_threshold=0.002,
                ransac_n=3,
                num_iterations=500
            )
            inlier_ratio = len(inliers) / len(cluster_points)
            print(f"Cluster {i}: Plane inlier ratio = {inlier_ratio:.2f}")

            # Smaller inlier ratio → better plane? (depends on scoring)
            if inlier_ratio < best_plane_score:
                checkerboard_points = np.asarray(pcd_cluster.select_by_index(inliers).points)
                best_plane_score = inlier_ratio
                best_plane_idx = i
                checkerboard_color = colors[i]  # keep cluster color

        except Exception as e:
            print(f"Cluster {i} plane fitting failed: {e}")

    # -----------------------------
    # 5. Visualize checkerboard plane with original cluster color
    # -----------------------------
    # Compute mean nearest-neighbor spacing immediately
    nbrs = NearestNeighbors(n_neighbors=2).fit(checkerboard_points)
    distances, _ = nbrs.kneighbors(checkerboard_points)
    mean_cell_recon = np.mean(distances[:, 1])
    print(f"Mean reconstructed cell spacing: {mean_cell_recon:.6f} (arbitrary units)")
    
    checkerboard_pcd = o3d.geometry.PointCloud()
    checkerboard_pcd.points = o3d.utility.Vector3dVector(checkerboard_points)
    checkerboard_pcd.paint_uniform_color(checkerboard_color)  # use original cluster color

    o3d.visualization.draw_geometries(
        [checkerboard_pcd],
        window_name=f"Detected Checkerboard Plane (Original Cluster Color)"
    )


    # # -----------------------------
    # # 4. Identify checkerboard candidate by planarity
    # # -----------------------------
    # best_plane_score = float("inf")
    # checkerboard_points = None

    # for i, (cluster_points, pcd_cluster) in enumerate(clusters):
    #     if len(cluster_points) < 10:
    #         continue  # ignore tiny clusters (noise)

    #     # Fit plane to each cluster
    #     try:
    #         plane_model, inliers = pcd_cluster.segment_plane(
    #             distance_threshold=0.002,
    #             ransac_n=3,
    #             num_iterations=500
    #         )
    #         # Ratio for how many points fitted into plane 
    #         inlier_ratio = len(inliers) / len(cluster_points)
    #         print(f"Cluster {i}: Plane inlier ratio = {inlier_ratio:.2f}")

    #         # Higher inlier ratio = more planar → likely checkerboard
    #         # TODO Need justification for 0.8? - inlier_ratio > 0.8 and 
    #         # Just take relative best one
    #         if inlier_ratio < best_plane_score:
    #             checkerboard_points = np.asarray(pcd_cluster.select_by_index(inliers).points)
    #             best_plane_score = inlier_ratio
    #             best_plane_idx = i
    #     except Exception as e:
    #         print(f"Cluster {i} plane fitting failed: {e}")

    # # TODO delete
    # if checkerboard_points is None:
    #     print("No strong planar checkerboard cluster found. Try adjusting RANSAC parameters.")
    #     exit()

    # print(f"Selected Cluster {best_plane_idx} as checkerboard candidate ({checkerboard_points.shape[0]} pts)")

    # # -----------------------------
    # # 5. Compute mean nearest-neighbor spacing (reconstructed cell size)
    # # -----------------------------
    # nbrs = NearestNeighbors(n_neighbors=2).fit(checkerboard_points)
    # distances, _ = nbrs.kneighbors(checkerboard_points)
    # mean_cell_recon = np.mean(distances[:, 1])
    # print(f"Mean reconstructed cell spacing: {mean_cell_recon:.6f} (arbitrary units)")

    # # Visualize checkerboard plane
    # checkerboard_pcd = o3d.geometry.PointCloud()
    # checkerboard_pcd.points = o3d.utility.Vector3dVector(checkerboard_points)
    # checkerboard_pcd.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries(
    #     [checkerboard_pcd],
    #     window_name="Detected Checkerboard Plane (Blue)"
    # )

    # -----------------------------
    # 6. Compute scale factor from checkerboard
    # -----------------------------
    scale_factor = cb.compute_scale(np.array([[0, 0, 0], [mean_cell_recon, 0, 0]]))
    print(f"Computed scale factor: {scale_factor:.6f}")

    # -----------------------------
    # 7. Apply scale factor to entire point cloud
    # -----------------------------
    points_scaled = points * scale_factor
    scaled_pcd = o3d.geometry.PointCloud()
    scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
    print("Applied metric scaling to point cloud.")

    # -----------------------------
    # 8. Visualize scaled result (checkerboard + object)
    # -----------------------------
    scaled_checkerboard = o3d.geometry.PointCloud()
    scaled_checkerboard.points = o3d.utility.Vector3dVector(checkerboard_points * scale_factor)
    scaled_checkerboard.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries(
        [scaled_pcd, scaled_checkerboard],
        window_name="Scaled Cloud (Blue = Checkerboard)"
    )



# # K means clustering - 1 cluster was satellite body + solar panel, other cluster was checkerboard and solar panel
# if __name__ == "__main__":
#     import open3d as o3d
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans
#     from sklearn.neighbors import NearestNeighbors

#     # -----------------------------
#     # 1. Calibrate camera using checkerboard images
#     # -----------------------------
#     cb = Checkerboard()
#     cb.read_ims("images/sat_checkerboard_pgm")
#     cb.undistort_ims(grid_size=(3, 3), cell_size=0.016)  # 4x4 internal corners, 16 mm cells
#     cb.plot_checkerboards()

#     # -----------------------------
#     # 2. Load point cloud (object + checkerboard)
#     # -----------------------------
#     pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
#     points = np.asarray(pcd.points)
#     print(f"Loaded point cloud with {points.shape[0]} points")

#     # -----------------------------
#     # 3. Segment point cloud using K-means clustering
#     # -----------------------------
#     kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(points)
#     labels = kmeans.labels_

#     # Separate clusters
#     cluster1 = points[labels == 0]
#     cluster2 = points[labels == 1]
#     print(f"Cluster 1: {cluster1.shape[0]} points")
#     print(f"Cluster 2: {cluster2.shape[0]} points")

#     # Visualize both clusters (red vs green)
#     pcd_cluster1 = o3d.geometry.PointCloud()
#     pcd_cluster1.points = o3d.utility.Vector3dVector(cluster1)
#     pcd_cluster1.paint_uniform_color([1, 0, 0])  # red

#     pcd_cluster2 = o3d.geometry.PointCloud()
#     pcd_cluster2.points = o3d.utility.Vector3dVector(cluster2)
#     pcd_cluster2.paint_uniform_color([0, 1, 0])  # green

#     o3d.visualization.draw_geometries(
#         [pcd_cluster1, pcd_cluster2],
#         window_name="K-means clusters (Red vs Green)"
#     )

#     # -----------------------------
#     # 4. Pick smaller cluster as checkerboard
#     # -----------------------------
#     if cluster1.shape[0] < cluster2.shape[0]:
#         checkerboard_points = cluster1
#         object_points = cluster2
#     else:
#         checkerboard_points = cluster2
#         object_points = cluster1

#     print(f"Selected checkerboard cluster with {checkerboard_points.shape[0]} points")

#     # -----------------------------
#     # 5. Compute reconstructed cell size (mean nearest neighbor distance)
#     # -----------------------------
#     nbrs = NearestNeighbors(n_neighbors=2).fit(checkerboard_points)
#     distances, indices = nbrs.kneighbors(checkerboard_points)
#     mean_cell_recon = np.mean(distances[:, 1])  # skip self-distance
#     print(f"Mean reconstructed cell size: {mean_cell_recon:.6f} (arbitrary units)")

#     # -----------------------------
#     # 6. Visualize nearest-neighbor connections on checkerboard
#     # -----------------------------
#     lines = [[i, indices[i, 1]] for i in range(len(checkerboard_points))]
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(checkerboard_points),
#         lines=o3d.utility.Vector2iVector(lines),
#     )
#     line_set.paint_uniform_color([0, 0, 1])  # blue lines

#     checkerboard_pcd = o3d.geometry.PointCloud()
#     checkerboard_pcd.points = o3d.utility.Vector3dVector(checkerboard_points)
#     checkerboard_pcd.paint_uniform_color([0, 0, 1])  # blue
#     object_pcd = o3d.geometry.PointCloud()
#     object_pcd.points = o3d.utility.Vector3dVector(object_points)
#     object_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # gray

#     o3d.visualization.draw_geometries(
#         [checkerboard_pcd, line_set],
#         window_name="Nearest Neighbor Links (Scale Estimation)"
#     )

#     # -----------------------------
#     # 7. Compute scale factor using Checkerboard class
#     # -----------------------------
#     scale_factor = cb.compute_scale(np.array([[0, 0, 0], [mean_cell_recon, 0, 0]]))
#     print(f"Computed scale factor from checkerboard: {scale_factor:.6f}")

#     # -----------------------------
#     # 8. Apply scale to entire point cloud
#     # -----------------------------
#     points_scaled = points * scale_factor
#     scaled_pcd = o3d.geometry.PointCloud()
#     scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
#     print("Applied metric scale to point cloud")

#     # -----------------------------
#     # 9. Remove checkerboard points (keep object only)
#     # -----------------------------
#     object_scaled_points = object_points * scale_factor
#     object_only_pcd = o3d.geometry.PointCloud()
#     object_only_pcd.points = o3d.utility.Vector3dVector(object_scaled_points)

#     print(f"Remaining points (object only): {object_scaled_points.shape[0]}")

#     # -----------------------------
#     # 10. Visualize scaled object (checkerboard removed)
#     # -----------------------------
#     o3d.visualization.draw_geometries(
#         [object_only_pcd],
#         window_name="Scaled Object (Checkerboard Removed)"
#     )


# RANSAC
# if __name__ == "__main__":
#     import open3d as o3d
#     import numpy as np
#     from sklearn.neighbors import NearestNeighbors
#     import matplotlib.pyplot as plt

#     # -----------------------------
#     # 1. Calibrate camera using checkerboard images
#     # -----------------------------
#     cb = Checkerboard() 
#     cb.read_ims("images/sat_checkerboard_pgm") 
#     cb.undistort_ims(grid_size=(3, 3), cell_size=0.016)  # 4x4 internal corners, 16 mm cells
#     cb.plot_checkerboards()

#     # -----------------------------
#     # 2. Load point cloud (object + checkerboard)
#     # -----------------------------
#     pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
#     points = np.asarray(pcd.points)
#     print(f"Loaded point cloud with {points.shape[0]} points")

#     # -----------------------------
#     # 3. Detect checkerboard plane automatically using RANSAC
#     # -----------------------------
#     plane_model, inliers = pcd.segment_plane(
#         distance_threshold=0.002,  # adjust if needed
#         ransac_n=3,
#         num_iterations=1000
#     )

#     # Separate inliers (checkerboard) and outliers (everything else)
#     checkerboard_pcd = pcd.select_by_index(inliers)
#     checkerboard_points = np.asarray(checkerboard_pcd.points)
#     object_only_pcd = pcd.select_by_index(inliers, invert=True)
#     print(f"Detected {checkerboard_points.shape[0]} points on checkerboard plane")

#     # -----------------------------
#     # Visualize the detected plane vs rest of point cloud
#     # -----------------------------
#     checkerboard_colored = checkerboard_pcd.paint_uniform_color([1, 0, 0])  # red = checkerboard
#     object_colored = object_only_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # grey = object
#     o3d.visualization.draw_geometries(
#         [checkerboard_colored, object_colored],
#         window_name="Detected Checkerboard Plane (Red)"
#     )

#     # -----------------------------
#     # 4. Compute reconstructed cell size (mean nearest neighbor distance)
#     # -----------------------------
#     nbrs = NearestNeighbors(n_neighbors=2).fit(checkerboard_points)
#     distances, indices = nbrs.kneighbors(checkerboard_points)
#     mean_cell_recon = np.mean(distances[:, 1])  # skip self-distance
#     print(f"Mean reconstructed cell size: {mean_cell_recon:.6f} (arbitrary units)")

#     # Visualize nearest-neighbor connections on checkerboard
#     lines = []
#     for i in range(len(checkerboard_points)):
#         neighbor_idx = indices[i, 1]
#         lines.append([i, neighbor_idx])
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(checkerboard_points),
#         lines=o3d.utility.Vector2iVector(lines),
#     )
#     line_set.paint_uniform_color([0, 0, 1])  # blue lines = neighbor pairs
#     o3d.visualization.draw_geometries(
#         [checkerboard_colored, line_set],
#         window_name="Nearest Neighbor Links (for Scale Estimation)"
#     )

#     # -----------------------------
#     # 5. Compute scale factor using Checkerboard method
#     # -----------------------------
#     scale_factor = cb.compute_scale(np.array([[0, 0, 0], [mean_cell_recon, 0, 0]]))
#     print(f"Computed scale factor from checkerboard: {scale_factor:.6f}")

#     # -----------------------------
#     # 6. Apply scale to entire point cloud
#     # -----------------------------
#     points_scaled = points * scale_factor
#     pcd.points = o3d.utility.Vector3dVector(points_scaled)
#     object_only_pcd = pcd.select_by_index(inliers, invert=True)
#     print(f"Applied metric scale to point cloud")

#     # -----------------------------
#     # 7. Visualize scaled object
#     # -----------------------------
#     o3d.visualization.draw_geometries(
#         [object_only_pcd],
#         window_name="Scaled Object (Checkerboard Removed)"
#     )



# # Original + scale
# if __name__ == "__main__": 
#     cb = Checkerboard() 
#     cb.read_ims("images/sat_checkerboard_pgm") 
#     # cb.undistort_ims(grid_size=(8,6), cell_size=0.116)
#     cb.undistort_ims(grid_size=(3, 3), cell_size=0.016)
#     cb.plot_checkerboards() 

#     # Example reconstructed checkerboard points (arbitrary scale)
#     reconstructed_points = np.random.rand(48, 3)  # just an example placeholder

#     # Compute and apply scale
#     scale = cb.compute_scale(reconstructed_points)
#     reconstructed_points_scaled = cb.apply_scale(reconstructed_points, scale)
        
#     plt.show()


 
