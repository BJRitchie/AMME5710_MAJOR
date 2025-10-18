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

    
    # def plot_checkerboards(self):
    #     """
    #     Visualize the checkerboard poses estimated from calibration in 3D camera space.
        
    #     Args:
    #         grid_size (tuple): Number of inner corners (cols, rows) used in calibration.
    #         cell_size (float): Physical size of each cell in meters.
    #     """

    #     assert hasattr(self, "_grid_size") and hasattr(self, "_cell_size"), \
    #         "Grid size and cell size not defined."

    #     assert hasattr(self, "_rvecs") and hasattr(self, "_tvecs"), \
    #         "Run undistort_ims() first to compute rvecs and tvecs."
        
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Draw camera origin
    #     ax.scatter(0, 0, 0, color='red', s=80, label='Camera Origin')
    #     ax.text(0, 0, 0, 'Camera', color='red')

    #     # Prepare board corner points (same as during calibration)
    #     objp = np.zeros((np.prod(self._grid_size), 3), np.float32)
    #     objp[:, :2] = np.indices(self._grid_size).T.reshape(-1, 2)
    #     objp *= self._cell_size

    #     for i, (rvec, tvec) in enumerate(zip(self._rvecs, self._tvecs)):
    #         R, _ = cv2.Rodrigues(rvec)
    #         board_points = (R @ objp.T + tvec).T

    #         # Plot checkerboard corners
    #         ax.scatter(board_points[:, 0], board_points[:, 1], board_points[:, 2], s=10, label=f'Checkerboard {i+1}')

    #         # Draw coordinate axes of each board (x=red, y=green, z=blue)
    #         origin = tvec.flatten()
    #         axis_length = self._cell_size * self._grid_size[0] / 2
    #         axes = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])
    #         axes_transformed = (R @ axes.T + origin.reshape(3, 1)).T

    #         ax.plot([origin[0], axes_transformed[0, 0]], [origin[1], axes_transformed[0, 1]], [origin[2], axes_transformed[0, 2]], color='r')
    #         ax.plot([origin[0], axes_transformed[1, 0]], [origin[1], axes_transformed[1, 1]], [origin[2], axes_transformed[1, 2]], color='g')
    #         ax.plot([origin[0], axes_transformed[2, 0]], [origin[1], axes_transformed[2, 1]], [origin[2], axes_transformed[2, 2]], color='b')

    #     # Plot settings
    #     ax.set_xlabel('X [m]')
    #     ax.set_ylabel('Y [m]')
    #     ax.set_zlabel('Z [m]')
    #     ax.set_title('Checkerboard Poses in Camera Frame')
    #     ax.legend()
    #     ax.view_init(elev=25, azim=35)
    #     ax.grid(True)
    #     plt.tight_layout()
    #     plt.show()










if __name__ == "__main__": 
    cb = Checkerboard() 
    cb.read_ims("images/checkerboards") 
    cb.undistort_ims(grid_size=(8,6), cell_size=0.116)
    cb.plot_checkerboards() 
    
    plt.show()


 
