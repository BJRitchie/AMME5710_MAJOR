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

if __name__ == "__main__":
    import open3d as o3d
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt

    # -----------------------------
    # 1. Calibrate camera using checkerboard images
    # -----------------------------
    cb = Checkerboard() 
    cb.read_ims("images/sat_checkerboard_pgm") 
    cb.undistort_ims(grid_size=(3, 3), cell_size=0.016)  # 4x4 internal corners, 16 mm cells
    cb.plot_checkerboards()

    # -----------------------------
    # 2. Load point cloud (object + checkerboard)
    # -----------------------------
    pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
    points = np.asarray(pcd.points)
    print(f"Loaded point cloud with {points.shape[0]} points")

    # -----------------------------
    # 3. Detect checkerboard plane automatically using RANSAC
    # -----------------------------
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.002,  # adjust if needed
        ransac_n=3,
        num_iterations=1000
    )

    # Separate inliers (checkerboard) and outliers (everything else)
    checkerboard_pcd = pcd.select_by_index(inliers)
    checkerboard_points = np.asarray(checkerboard_pcd.points)
    object_only_pcd = pcd.select_by_index(inliers, invert=True)
    print(f"Detected {checkerboard_points.shape[0]} points on checkerboard plane")

    # -----------------------------
    # Visualize the detected plane vs rest of point cloud
    # -----------------------------
    checkerboard_colored = checkerboard_pcd.paint_uniform_color([1, 0, 0])  # red = checkerboard
    object_colored = object_only_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # grey = object
    o3d.visualization.draw_geometries(
        [checkerboard_colored, object_colored],
        window_name="Detected Checkerboard Plane (Red)"
    )

    # -----------------------------
    # 4. Compute reconstructed cell size (mean nearest neighbor distance)
    # -----------------------------
    nbrs = NearestNeighbors(n_neighbors=2).fit(checkerboard_points)
    distances, indices = nbrs.kneighbors(checkerboard_points)
    mean_cell_recon = np.mean(distances[:, 1])  # skip self-distance
    print(f"Mean reconstructed cell size: {mean_cell_recon:.6f} (arbitrary units)")

    # Visualize nearest-neighbor connections on checkerboard
    lines = []
    for i in range(len(checkerboard_points)):
        neighbor_idx = indices[i, 1]
        lines.append([i, neighbor_idx])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(checkerboard_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0, 0, 1])  # blue lines = neighbor pairs
    o3d.visualization.draw_geometries(
        [checkerboard_colored, line_set],
        window_name="Nearest Neighbor Links (for Scale Estimation)"
    )

    # -----------------------------
    # 5. Compute scale factor using Checkerboard method
    # -----------------------------
    scale_factor = cb.compute_scale(np.array([[0, 0, 0], [mean_cell_recon, 0, 0]]))
    print(f"Computed scale factor from checkerboard: {scale_factor:.6f}")

    # -----------------------------
    # 6. Apply scale to entire point cloud
    # -----------------------------
    points_scaled = points * scale_factor
    pcd.points = o3d.utility.Vector3dVector(points_scaled)
    object_only_pcd = pcd.select_by_index(inliers, invert=True)
    print(f"Applied metric scale to point cloud")

    # -----------------------------
    # 7. Visualize scaled object
    # -----------------------------
    o3d.visualization.draw_geometries(
        [object_only_pcd],
        window_name="Scaled Object (Checkerboard Removed)"
    )



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


 
