import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import pycolmap 
import os 

from sfm_pipeline_lib import StrcFromMotion 
from checkerboard_lib import Checkerboard

class PointCloudMatcher: 
    def __init__(self, sfm: StrcFromMotion, cb: Checkerboard):
        
        # Store the classes 
        self._SFM = copy.deepcopy(sfm)
        self._CB = copy.deepcopy(cb) 
        
        # Internal variables
        self._R = None 
        self._t = None 
        self._T = None # transformation matrix [R|t]
        self._s = None # scale 
        
        # Pose information   
        self._poses0 = None
        self._poses1 = None
        self._orientations0 = None
        self._orientations1 = None

        # Pointclouds 
        self._pc0 = None 
        self._pc1 = None  

        pass
    
    def matchPoints( self, sparse_path = "sparse/0" ): 
        
        # Checkerboard camera poses 
        checker_rotations, checker_translations = self._CB.get_camera_poses()
        checker_translations = np.array(checker_translations)[:, :, 0].T # turn into 3xN
        checker_rotations = np.array(checker_rotations).transpose(1, 2, 0) # turn into 3xN

        # Get SFM camera poses 
        sfm_rotations, sfm_translations, pcd = self._SFM.get_pointcloud_and_poses()
        sfm_pcd_np = np.asarray(pcd.points).T 
        sfm_translations = np.array(sfm_translations)[:, :, 0].T 
        sfm_rotations = np.array(sfm_rotations).transpose(1, 2, 0)

        # Extract the matched poses 
        matched_indices_sfm, matched_indices_cb = self._match_sfm_camera_poses( sparse_path = sparse_path )
        sfm_translations_matched    = (sfm_translations[:, matched_indices_sfm])
        sfm_rotations_matched       = (sfm_rotations[:, :, matched_indices_sfm])
        checker_trans_matched       = (checker_translations[:, matched_indices_cb])
        checker_rot_matched         = (checker_rotations[:, :, matched_indices_cb])

        # Apply matching to poses 
        R, t, s, T, best_inliers = self.matchFromPosesRANSAC( 
            poses0=sfm_translations_matched, 
            poses1=checker_trans_matched, 
            threshold=0.05, 
            ransac_samples=5 ) 

        print(f"Number of inliers: { len(best_inliers) }")

        print("\n====== Umeyama ======") 
        print("R: ") 
        print(R)
        print("\nt: ") 
        print(t)
        print("\ns: ") 
        print(s)

        self.transformPointClouds( sfm_pcd_np, None ) 
        self.transformCameraPoses( 
            sfm_translations_matched, sfm_rotations_matched, checker_trans_matched, checker_rot_matched )

        return R, t, s, T 
            
    def matchFromPosesRANSAC( self, poses0, poses1, threshold=0.02, max_iter=1000, ransac_samples=5 ): 
        
        N = poses0.shape[1]
        best_inliers = [] 

        # Try fit using samples, check for quality of fit (RANSAC)
            # Keep the best fit
        for _ in range(max_iter):
            # Random 3-point subset
            idx = np.random.choice(N, ransac_samples, replace=False)
            R, t, s, T = self.matchFromPosesUmeyama(poses0[:, idx], poses1[:, idx])

            # Compute alignment error for all points
            transformed = np.empty_like(poses0)
            for i in range(N):
                transformed[:, i] = s * (R @ poses0[:, i]) + t 
                
            errors = np.linalg.norm(transformed - poses1, axis=0)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        # Refine using all inliers
        if len(best_inliers) >= 3:
            R, t, s, T = self.matchFromPosesUmeyama(poses0[:, best_inliers], poses1[:, best_inliers])
        else:
            R, t, s, T = self.matchFromPosesUmeyama(poses0, poses1)

        return R, t, s, T, best_inliers 
    
    def matchFromPosesUmeyama( self, poses0, poses1 ): 
        """
        Solve for rotation, translation, and scale between two sets of 3D points.

        Uses a linear least-squares (Umeyama) method to find the similarity transform:
            poses1 ≈ s * R @ poses0 + t

        Args:
            poses0 (np.ndarray): 3xN array of points in frame 0
            poses1 (np.ndarray): 3xN array of points in frame 1

        Returns:
            R (np.ndarray): 3x3 rotation matrix
            t (np.ndarray): 3x1 translation vector
            s (float): uniform scale factor
            T (np.ndarray): 4x4 homogeneous transformation matrix
        """
        
        # --- 1. Input checks
        poses0 = np.asarray(poses0)
        poses1 = np.asarray(poses1)
        
        # Store 
        self._poses0 = poses0
        self._poses1 = poses1

        if poses0.shape != poses1.shape:
            raise ValueError("poses0 and poses1 must have the same shape")
        if poses0.shape[0] != 3:
            raise ValueError("Input arrays must be 3xN")

        N = poses0.shape[1]

        # --- 2. Compute centroids
        mu0 = np.mean(poses0, axis=1, keepdims=True)
        mu1 = np.mean(poses1, axis=1, keepdims=True)

        # --- 3. Center the data
        X0 = poses0 - mu0
        X1 = poses1 - mu1

        # Compute covariance matrix
        Sigma = (X1 @ X0.T) / N

        # Compute SVD of covariance
        U, D, Vt = np.linalg.svd(Sigma)
        
        # Two possible rotations: proper (+1) and reflected (-1)
        S_pos = np.eye(3)
        S_neg = np.diag([1, 1, -1])

        R_pos = U @ S_pos @ Vt
        R_neg = U @ S_neg @ Vt
        
        # Compute scales
        var0 = np.sum(X0**2) / N
        s_pos = np.trace(np.diag(D) @ S_pos) / var0
        s_neg = np.trace(np.diag(D) @ S_neg) / var0

        # Compute translations
        t_pos = mu1 - s_pos * R_pos @ mu0
        t_neg = mu1 - s_neg * R_neg @ mu0

        # Compute total alignment error for both
        pos_t = np.empty_like(poses0)
        neg_t = np.empty_like(poses0)
        for i in range(N):
            pos_t[:, i] = s_pos * (R_pos @ poses0[:, i]) + t_pos[:, 0]
            neg_t[:, i] = s_neg * (R_neg @ poses0[:, i]) + t_neg[:, 0]
        err_pos = np.mean(np.linalg.norm(pos_t - poses1, axis=0))
        err_neg = np.mean(np.linalg.norm(neg_t - poses1, axis=0))

        # Choose the better one
        if err_pos <= err_neg:
            R, t, s = R_pos, t_pos, s_pos
        else:
            R, t, s = R_neg, t_neg, s_neg

        # Build homogeneous matrix
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t.flatten() 

        # Store 
        self._R = R 
        self._t = t.flatten() 
        self._T = T  
        self._s = s 
        
        return R, t.flatten(), s, T

    def matchFromPosesOrientsRANSAC( self, poses0, rots0, poses1, rots1, w_rot=1.0, max_iter=1000, ransac_samples =5, threshold=0.02 ): 
        N = poses0.shape[1]
        best_inliers = [] 

        # Try fit using samples, check for quality of fit (RANSAC)
            # Keep the best fit
        for _ in range(max_iter):
            # Random 3-point subset
            idx = np.random.choice(N, ransac_samples, replace=False)
            R, t, s, T = self.matchFromPosesWithOrientation(
                poses0[:, idx], rots0[:, :, idx], poses1[:, idx], rots1[:, :, idx]) 

            # Compute alignment error for all points
            # transformed = np.empty_like(poses0)
            # for i in range(N):
            #     transformed[:, i] = s * (R @ poses0[:, i]) + t 
            
            # errors = np.linalg.norm(transformed - poses1, axis=0)
            errors = self._residuals( poses0, rots0, poses1, rots1, s, R, t, w_rot )
            inliers = np.where(errors < threshold)[0] 

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        # Refine using all inliers
        if len(best_inliers) >= 3:
            R, t, s, T = self.matchFromPosesWithOrientation(
                poses0[:, best_inliers], rots0[:, :, best_inliers], 
                poses1[:, best_inliers], rots1[:, :, best_inliers], w_rot=w_rot) 
        else:
            print("Warning: Too few inliers. ")
            R, t, s, T = self.matchFromPosesWithOrientation(
                poses0, rots0, poses1, rots1, w_rot=w_rot) 

        return R, t, s, T, best_inliers 

    def matchFromPosesWithOrientation(self, poses0, rots0, poses1, rots1, w_rot=1.0):
        """
        Estimate similarity transform (s, R, t) using both camera positions and orientations.
        Minimizes combined translation + orientation misalignment.
        """
        import cv2 
        
        # --- Step 1: Get initial guess from Umeyama (positions only)
        R_init, t_init, s_init, T_init = self.matchFromPosesUmeyama(poses0, poses1)

        # --- Step 2: Refine with orientation cost (optional)
        def residual(params):
            s = params[0]
            rvec = params[1:4]
            t = params[4:7]
            
            Rg, _ = cv2.Rodrigues(rvec)
            
            # Transform positions
            # transformed = s * (Rg @ poses0) + t.reshape(3,1)
            N = poses0.shape[1] 
            transformed = np.empty_like(poses0) 
            for i in range(N):
                transformed[:, i] = s * (Rg @ poses0[:, i]) - t
            pos_err = np.linalg.norm(transformed - poses1, axis=0).mean()
            
            # Orientation error: angle between Rg @ R0 and R1
            rot_errs = []
            for i in range(rots0.shape[2]):
                R_pred = Rg @ rots0[:, :, i]
                dR = R_pred.T @ rots1[:, :, i]
                angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1, 1))
                rot_errs.append(angle)
                
            rot_err = np.mean(rot_errs)
            return pos_err + w_rot * rot_err

        # Minimise the orientation cost 
        from scipy.optimize import minimize
        x0 = np.hstack([s_init, np.zeros(3), t_init])
        res = minimize(residual, x0, method='Powell')

        s = res.x[0]
        rvec = res.x[1:4]
        t = res.x[4:7] 
        Rg, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3, :3] = s * Rg
        T[:3, 3] = t

        return Rg, t, s, T

    def transformPointClouds( self, pc0, pc1  ): 
        """Transforms pointcloud 0 into the same frame of reference as pointcloud 1. 

        Args:
            pc0 (np.array): Pointcloud 0, shape (3xN)
            pc1 (np.array): Target Pointcloud, shape (3xN)

        """
        assert (self._R.any() != None) and (self._t.any() != None), "Rotation and translation vectors are not defined. "
        
        # --- Input checks
        # pc0 = np.asarray(pc0)

        if pc0.shape[0] != 3:
            raise ValueError("Input arrays must be 3xN")
        
        N = pc0.shape[1]
        pc0_t = np.empty_like(pc0)
        
        # Perform transformation 
        # pc0_t = self._s * (self._R @ pc0) + self._t
        
        for i in range(N):
            pnt = pc0[:, i]
            pc0_t[:, i] = self._s * (self._R @ pnt) + self._t

        # Store 
        self._pc0 = self._s * pc0 # scale to size 
        self._pc1 = pc1
        self._pc0_t = pc0_t 

        return pc0_t
    
    def transformCameraPoses(self, cam_centers0, cam_rots0, cam_centers1, cam_rots1):
        """
        Transform camera positions and orientations from frame 0 to frame 1.

        Args:
            cam_centers0 (np.ndarray): 3xN camera centers (positions) in frame 0
            cam_rots0 (np.ndarray): 3x3xN rotation matrices (orientations) in frame 0

        Returns:
            cam_centers1 (np.ndarray): transformed camera centers in frame 1
            cam_rots1 (np.ndarray): transformed rotation matrices in frame 1
        """
        assert self._R is not None and self._t is not None and self._s is not None, "Transform not defined"

        N = cam_centers0.shape[1]
        cam_centers_t = np.zeros_like(cam_centers0)
        cam_rots_t = np.zeros_like(cam_rots0)

        for i in range(N):
            c0 = cam_centers0[:, i]
            R0 = cam_rots0[:, :, i]
            cam_centers_t[:, i] = self._s * (self._R @ c0) + self._t
            cam_rots_t[:, :, i] = self._R @ R0

        # Store internally 
        self._cam_centers0 = self._s * cam_centers0 # scale to same size as others 
        self._cam_rots0 = cam_rots0 
        self._cam_centers0_t = cam_centers_t 
        self._cam_rots0_t = cam_rots_t 
        self._cam_centers1 = cam_centers1 
        self._cam_rots1 = cam_rots1 
        
        return cam_centers_t, cam_rots_t
    
    def plotMultiPointClouds(self, camera_scale=0.1, show_frustums=True, plot_original=False):
        """
        Visualize original, transformed, and target point clouds along with their camera poses.
        Uses consistent world-space transformation logic as in the COLMAP plotting function.

        Colors:
            - Green  = Original pc0 and cameras (Frame 0)
            - Red    = Transformed pc0_t and cameras (Frame 0 transformed)
            - Blue   = Target pc1 and cameras (Frame 1)
        """
        import open3d as o3d
        import numpy as np

        assert self._pc0 is not None and self._pc0_t is not None, "Missing point clouds."
        assert hasattr(self, "_cam_centers0") and hasattr(self, "_cam_centers1"), "Camera poses not defined."

        r = [1.0, 0.0, 0.0]
        g = [0.0, 1.0, 0.0]
        b = [0.0, 0.0, 1.0]

        geometries = []

        # --- Convert arrays to Open3D point clouds ---
        pc0_orig = o3d.geometry.PointCloud()
        pc0_trans = o3d.geometry.PointCloud()

        pc0_orig.points = o3d.utility.Vector3dVector(self._pc0.T)
        pc0_trans.points = o3d.utility.Vector3dVector(self._pc0_t.T)

        pc0_orig.paint_uniform_color(g)
        pc0_trans.paint_uniform_color(r)
        
        if plot_original: 
            geometries.extend([pc0_orig, pc0_trans])
        else: 
            geometries.append(pc0_trans)

        if type(self._pc1) != type(None): 
            pc1 = o3d.geometry.PointCloud()
            pc1.points = o3d.utility.Vector3dVector(self._pc1.T)
            pc1.paint_uniform_color(b) 
            geometries.append(pc1)

        # --- Helper to create frustum in world space ---
        def make_camera_frustum(center, R_world_from_cam, color, scale):
            """Create a small camera coordinate frame + pyramid frustum in world space."""
            geoms = []

            # Camera frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
            T = np.eye(4)
            T[:3, :3] = R_world_from_cam
            T[:3, 3] = center
            frame.transform(T)
            frame.paint_uniform_color(color)
            geoms.append(frame)

            if show_frustums:
                # Define a simple frustum in camera coordinates
                depth = scale * 2
                w, h = scale, scale * 0.75
                corners_cam = np.array([
                    [0, 0, 0],
                    [-w, -h, depth],
                    [ w, -h, depth],
                    [ w,  h, depth],
                    [-w,  h, depth],
                ])
                # Transform to world space
                corners_world = (R_world_from_cam @ corners_cam.T).T + center

                lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corners_world)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
                geoms.append(line_set)

            return geoms

        # --- Plot all camera sets ---
        def add_cameras(cam_centers, cam_rots, color):
            """Add a set of cameras to the geometry list."""
            N = cam_centers.shape[1]
            for i in range(N):
                c = cam_centers[:, i]
                R = cam_rots[:, :, i]
                # Convert world→camera to camera→world if necessary
                R_world_from_cam = R.T
                geometries.extend(make_camera_frustum(c, R_world_from_cam, color, camera_scale))

        # Original cameras (Frame 0)
        if plot_original: 
            add_cameras(self._cam_centers0, self._cam_rots0, g)

        # Transformed cameras (Frame 0 aligned to Frame 1)
        if hasattr(self, "_cam_centers0_t"):
            add_cameras(self._cam_centers0_t, self._cam_rots0_t, r)

        # Target/reference cameras (Frame 1)
        add_cameras(self._cam_centers1, self._cam_rots1, b)

        print(f"Visualizing {self._cam_centers0.shape[1]} + {self._cam_centers1.shape[1]} cameras and {self._pc0.shape[1]} points per cloud.")
        print("Colors: green = original pc0, red = transformed pc0_t, blue = target pc1")
        o3d.visualization.draw_geometries(geometries)

    def savePointcloud( self, savename ): 
        
        return 

    def _residuals( self, poses0, rots0, poses1, rots1, s, R, t, w_rot=1.0 ): 
        # Transform positions
        N = poses0.shape[1] 
        transformed = np.empty_like(poses0)
        for i in range(N): 
            transformed[:, i] = s * (R @ poses0[:, i]) + t

        pos_err = np.linalg.norm(transformed - poses1, axis=0)
        
        # Orientation error: angle between Rg @ R0 and R1
        rot_errs = []
        for i in range(rots0.shape[2]):
            R_pred = R @ rots0[:, :, i]
            dR = R_pred.T @ rots1[:, :, i]
            angle = np.arccos(np.clip((np.trace(dR) - 1)/2, -1, 1))
            rot_errs.append(angle)
            
        rot_err = np.array(rot_errs) 
        err = pos_err + w_rot * rot_err
        return err 

    def _match_sfm_camera_poses( self, sparse_path ):
        
        rec = pycolmap.Reconstruction(sparse_path)
        images_sorted = sorted(rec.images.values(), key=lambda x: x.name)
        name_to_index = {os.path.basename(img.name): i for i, img in enumerate(images_sorted)}
        index_to_name = {i: os.path.basename(img.name) for i, img in enumerate(images_sorted)}

        matched_indices_sfm = []
        matched_indices_cb = []

        print("===== Matched Image Pairs =====")
        print(f"{'Checkerboard Image':40s}  |  {'SfM Image':40s}")

        for j, cb_name in enumerate(self._CB._checker_image_names):
            cb_base = os.path.basename(cb_name)
            
            if cb_base in name_to_index:
                sfm_idx = name_to_index[cb_base]
                sfm_name = index_to_name[sfm_idx]
                print(f"{cb_base:40s}  |  {sfm_name:40s}")

                matched_indices_sfm.append(sfm_idx)
                matched_indices_cb.append(j)

        print("\nTotal matched pairs:", len(matched_indices_sfm))
        return np.array(matched_indices_sfm, dtype=int), np.array(matched_indices_cb, dtype=int)



# test
if __name__ == "__main__": 
    import pickle 
    import os 
    from rotation_mat import rotation_mat_degs, apply_transform
    
    # ------- BASIC TEST ------- # 
    poses0_ = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 5.0, 6.0, 7.0],
        [7.0, 8.0, 9.0, 10.0]
    ])

    # Apply known transform: scale=2.0, rotation=90° about z, translation=[3, -2, 1]
    R_true_ = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
    s_true_ = 2.0
    t_true_ = np.array([[3], [-2], [1]])

    poses1_ = s_true_ * (R_true_ @ poses0_) + t_true_

    matcher_ = PointCloudMatcher()
    R_, t_, s_, T_ = matcher_.matchFromPosesRANSAC(poses0_, poses1_)

    print("Estimated scale:", s_)
    print("Estimated rotation:\n", R_)
    print("Estimated translation:\n", t_)
    print("\nHomogeneous transform:\n", T_)

    # Verify
    pred_ = s_ * (R_ @ poses0_) + t_.reshape(3,1)
    print("\nResidual error:\n", poses1_ - pred_)

    # ------- POINTCLOUD TEST ------- # 
    # File names 
    store_path_= "images/checker_nasa_box"
    vid_path_ = 'images/checker_nasa_box.mp4'
    sfm_save_path_ = "images/checker_nasa_box_sfm"

    # Storage files 
    im_path_ = store_path_
    db_path_ = "database.db"
    sparse_path_ = "sparse"
    dense_path_ = "dense"
    sat_model_path_ = "sat_model"

    
    # Load sfm class 
    with open("sfm_pipeline.pkl", "rb") as f_:
        sfm_pipeline_ = pickle.load(f_)

    with open("checkerboard.pkl", "rb") as f_:
        cb_ = pickle.load(f_)

    print("Loaded saved SfM pipeline and checkerboard data")

    # Construct paths
    store_name_ = os.path.join(sparse_path_, '0')
    file_path_ = os.path.join(store_name_, "points.ply")

    # Load point cloud
    if not os.path.exists(file_path_):
        raise FileNotFoundError(f"points.ply not found at {file_path_}")
    pcd_ = o3d.io.read_point_cloud(file_path_) # Output of SFM 

    # SFM points 
    sfm_pcd_ = np.asarray(pcd_.points).T 

    # Get SFM camera poses 
    sfm_rotations_, sfm_translations_ = sfm_pipeline_.get_poses()
    sfm_translations_ = np.array(sfm_translations_)[:, :, 0].T 
    sfm_rotations_ = np.array(sfm_rotations_).transpose(1, 2, 0)

    # Generate example transform 
    s_true_ = 2
    R_true_ = rotation_mat_degs(roll=45, pitch=72, yaw=132) 
    t_true_ = np.empty((3,1))
    t_true_[:,0] = [5, 10, 15]
    
    synth_translations_, synth_rotations_ = apply_transform(s_true_, R_true_, t_true_, sfm_translations_, sfm_rotations_)
    synth_pcd_, _ = apply_transform(s_true_, R_true_, t_true_, sfm_pcd_, None) 

    print("====== TRUE ======") 
    print("R_true: ") 
    print(R_true_)
    print("\nt_true: ") 
    print(t_true_)
    print("\ns_true: ") 
    print(s_true_)

    pc_matcher_ = PointCloudMatcher() 
    R_matcher_, t_matcher_, s_matcher_, T_matcher_ = pc_matcher_.matchFromPoses( sfm_translations_, synth_translations_ )

    print("\n====== Umeyama ======") 
    print("R: ") 
    print(R_matcher_)
    print("\nt: ") 
    print(t_matcher_)
    print("\ns: ") 
    print(s_matcher_)

    pc_matcher_.transformPointClouds( sfm_pcd_, synth_pcd_ )
    pc_matcher_.transformCameraPoses( sfm_translations_, sfm_rotations_, synth_translations_, synth_rotations_ )
    pc_matcher_.plotPointClouds()

