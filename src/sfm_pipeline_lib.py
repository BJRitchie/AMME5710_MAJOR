import pycolmap
import numpy as np
import cv2
import os
import shutil
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData
import copy

class StrcFromMotion: 
    def __init__(self, 
                 db_path:          str, 
                 im_path:          str, 
                 sparse_path:      str,
                 dense_path:       str,
                 sat_model_path:   str,
                 cam_mode:         pycolmap.CameraMode, 
                 cam_model:        str, 
                 reader_ops:       pycolmap.ImageReaderOptions, 
                 sift_ops:         pycolmap.SiftExtractionOptions, 
                 device:           pycolmap.Device): 
        
        # Paths 
        self._database_path     =db_path
        self._image_path        =im_path
        self._sparse_path       =sparse_path
        self._dense_path        =dense_path
        self._sat_model_path    =sat_model_path

        # Settings 
        self._camera_mode   =cam_mode
        self._camera_model  =cam_model
        self._reader_options=reader_ops
        self._sift_options  =sift_ops
        self._device        =device

        # Unassigned variables 
        self._image_names   =None 
        self._points_ply    =None
        
        # Clean up workspace 
        paths = [db_path, sparse_path, dense_path] 
        self._clean_up( paths )
        
        # Make fresh output directories 
        self._make_clean_dirs( [sparse_path, dense_path] )
        
        return 

    def resize_ims( self, im_path, max_size, interval=1 ): 
        
        # Get the image names 
        self._image_names = [f for f in os.listdir(im_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self._image_names = self._image_names[::interval] # Take only every Nth image 
        
        # Iterate over each file  
        for f in self._image_names:
            path = os.path.join(im_path, f)
            img = cv2.imread(path)
            
            if img is None:
                continue
            
            # Resize to fit common dimensions 
            h, w = img.shape[:2]
            scale = max_size / max(h, w) 
            if scale < 1.0:
                img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
                cv2.imwrite(path, img_resized)

    def prep_pointcloud( self ): 
        
        print("=== Extracting features ===")
        pycolmap.extract_features(
            database_path=  self._database_path,
            image_path=     self._image_path,
            image_names=    self._image_names,
            camera_mode=    self._camera_mode,
            camera_model=   self._camera_model,
            reader_options= self._reader_options,
            sift_options=   self._sift_options,
            device=         self._device 
        )
        
        print("=== Matching features (exhaustive) ===")
        pycolmap.match_exhaustive(
            database_path=          self._database_path,
            device=                 self._device,
            sift_options=           pycolmap.SiftMatchingOptions(),
            matching_options=       pycolmap.ExhaustiveMatchingOptions(),
            verification_options=   pycolmap.TwoViewGeometryOptions(),
        )

        print("=== Running incremental mapping ===")
        pycolmap.incremental_mapping(
            database_path=  self._database_path,
            image_path=     self._image_path,
            output_path=    self._sparse_path,
        )

        print("=== Undistorting images ===")
        pycolmap.undistort_images(
            output_path=    self._dense_path, 
            input_path=     os.path.join(self._sparse_path, "0"),  # first reconstruction
            image_path=     self._image_path,
            output_type=    "COLMAP",
        )

        return 

    def make_pointcloud( self, store_path = "0" ): 
        
        # Read the reconstructed point cloud 
        rec_path = os.path.join(self._sparse_path, store_path) 
        
        # Reconstruct the pointcloud 
        rec = pycolmap.Reconstruction(rec_path)
        
        print("Registered images:", len(rec.images))
        print("3D points:", len(rec.points3D))

        # Save the point cloud 
        self._points_ply = os.path.join(rec_path, "points.ply")
        rec.export_PLY(self._points_ply)
        print("Saved sparse point cloud to", self._points_ply)

        return 

    def clean_pointcloud( self, store_path="0", nb_points=50, radius=10.0 ):         
        # Read the pcloud 
        path = os.path.join( self._sparse_path, store_path, "points.ply" ) 
        pcd = o3d.io.read_point_cloud( path ) 
        
        # Apply radis outlier removal 
        cl, ind = pcd.remove_radius_outlier( nb_points = nb_points, radius = radius )
        
        # Extract the inlier point cloud
        denoised_pcd = pcd.select_by_index(ind) 
        
        # Store back in ply file 
        o3d.io.write_point_cloud(path, denoised_pcd, write_ascii=False, compressed=False) 

        return 
    
    def make_reference_ply( self, ref_path="reference.ply" ):
        '''
        Convert reference model from STL file to PLY format for visualization
        and surface matching.
        '''

        # Read the STL file
        stl_path = os.path.join( self._sat_model_path, "sat_model_stl.stl" )
        mesh = o3d.io.read_triangle_mesh( stl_path )

        # Convert to PLY format
        save_path = os.path.join( self._sat_model_path, ref_path )
        o3d.io.write_triangle_mesh( save_path, mesh, write_ascii=False, compressed=False )
        print(f"Saved reference model to {save_path}")

        return
    
    def surface_matching_ppf_icp(self, store_path="0", voxel_size=0.05, distance_threshold=0.02, 
                                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=2000):
        '''
        Perform surface matching between reference model and reconstructed point cloud
        using Point Pair Features (PPF) for initial alignment and ICP for refinement.
        
        Args:
            store_path: Path to the reconstructed point cloud
            voxel_size: Voxel size for downsampling
            distance_threshold: Distance threshold for ICP
            relative_fitness: Relative fitness threshold for ICP convergence
            relative_rmse: Relative RMSE threshold for ICP convergence
            max_iteration: Maximum ICP iterations
        '''
        print("=== Surface Matching: PPF + ICP ===")
        
        # Load reference model
        ref_path = os.path.join(self._sat_model_path, "reference.ply")
        if not os.path.exists(ref_path):
            print(f"Reference model not found at {ref_path}")
            return None
            
        ref_mesh = o3d.io.read_triangle_mesh(ref_path)
        
        # Check if mesh loaded properly
        if len(ref_mesh.vertices) == 0:
            print(f"Reference mesh has no vertices. Check file: {ref_path}")
            return None
        
        # Convert reference mesh to point cloud by sampling
        ref_pcd = ref_mesh.sample_points_uniformly(number_of_points=10000)
        print(f"Reference point cloud: {len(ref_pcd.points)} points")
        
        # Check if point cloud was generated properly
        if len(ref_pcd.points) == 0:
            print("Failed to generate point cloud from reference mesh")
            return None
        
        # Load reconstructed point cloud - try multiple possible locations
        possible_paths = [
            os.path.join(self._sat_model_path, "points.ply"), 
            os.path.join(store_path, "points.ply"),  # Direct path
            os.path.join(self._sparse_path, store_path, "points.ply"),  # Standard sparse reconstruction
        ]
        
        pcd_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pcd_path = path
                print(f"Found point cloud at: {pcd_path}")
                break
        
        if pcd_path is None:
            print(f"Reconstructed point cloud not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            return None
            
        target_pcd = o3d.io.read_point_cloud(pcd_path)
        print(f"Target point cloud: {len(target_pcd.points)} points")
        
        # Check if target point cloud loaded properly
        if len(target_pcd.points) == 0:
            print(f"Target point cloud has no points. Check file: {pcd_path}")
            return None
        
        # DIAGNOSTIC: Print point cloud properties
        print("\n=== DIAGNOSTIC INFORMATION ===")
        ref_bbox = ref_pcd.get_axis_aligned_bounding_box()
        target_bbox = target_pcd.get_axis_aligned_bounding_box()
        
        print(f"Reference model bounding box:")
        print(f"  Center: {ref_bbox.get_center()}")
        print(f"  Extent: {ref_bbox.get_extent()}")
        print(f"  Max extent: {np.max(ref_bbox.get_extent()):.4f}")
        
        print(f"Target point cloud bounding box:")
        print(f"  Center: {target_bbox.get_center()}")
        print(f"  Extent: {target_bbox.get_extent()}")
        print(f"  Max extent: {np.max(target_bbox.get_extent()):.4f}")
        
        # Calculate distance between centers
        center_distance = np.linalg.norm(ref_bbox.get_center() - target_bbox.get_center())
        print(f"Distance between centers: {center_distance:.4f}")
        
        # Scale ratio
        ref_scale = np.max(ref_bbox.get_extent())
        target_scale = np.max(target_bbox.get_extent())
        scale_ratio = ref_scale / target_scale if target_scale > 0 else float('inf')
        print(f"Scale ratio (ref/target): {scale_ratio:.4f}")
        print("==============================\n")
        
        # Apply initial scale and center normalization
        ref_pcd_normalized, target_pcd_normalized, scale_transform = self._normalize_point_clouds(
            ref_pcd, target_pcd, center_distance, scale_ratio)

        # Preprocess point clouds to remove any additional noise
        ref_pcd_processed, target_pcd_processed = self._preprocess_point_clouds(
            ref_pcd_normalized, target_pcd_normalized, voxel_size)
        
        # PPF-based initial alignment
        print("PPF-based initial alignment...")
        initial_transform = self._ppf_matching(ref_pcd_processed, target_pcd_processed, voxel_size)
        
        if initial_transform is None:
            print("PPF matching failed, using identity transformation")
            initial_transform = np.eye(4)
        
        # Apply initial transformation
        ref_pcd_aligned = ref_pcd_processed.transform(initial_transform)
        
        # ICP refinement to get precise alignment
        print("ICP refinement...")
        final_transform, fitness, rmse = self._icp_refinement(
            ref_pcd_aligned, target_pcd_processed, distance_threshold, 
            relative_fitness, relative_rmse, max_iteration)
        
        # Combine transformations
        combined_transform = final_transform @ initial_transform
        
        # Apply final transformation to normalized reference
        ref_pcd_final = ref_pcd_processed.transform(combined_transform)
        
        # Calculate adaptive distance threshold based on target point cloud scale
        target_extent = np.max(target_pcd_processed.get_axis_aligned_bounding_box().get_extent())
        adaptive_threshold = max(distance_threshold, target_extent * 0.05)  # At least 5% of object size
        print(f"Using adaptive distance threshold: {adaptive_threshold:.4f}")
        
        # Calculate alignment metrics
        metrics = self._calculate_alignment_metrics(ref_pcd_final, target_pcd_processed, adaptive_threshold)
        
        # Save results
        results = {
            'initial_transform': initial_transform,
            'final_transform': final_transform,
            'combined_transform': combined_transform,
            'icp_fitness': fitness,
            'icp_rmse': rmse,
            'alignment_metrics': metrics,
            'ref_pcd_aligned': ref_pcd_final,
            'target_pcd': target_pcd_processed,
            'scale_transform': scale_transform,
            'adaptive_threshold': adaptive_threshold
        }
        
        self._save_matching_results(results, store_path)
        self._visualize_matching_results(results)
        
        return results
    
    def _normalize_point_clouds(self, ref_pcd, target_pcd, center_distance, scale_ratio):
        '''
        Normalize point clouds to handle scale and translation differences
        '''
        print("Applying normalization to handle scale/translation differences...")
        
        # Create copies of point clouds
        ref_pcd_norm = copy.deepcopy(ref_pcd)
        target_pcd_norm = copy.deepcopy(target_pcd)
        
        # Get bounding boxes
        ref_bbox = ref_pcd.get_axis_aligned_bounding_box()
        target_bbox = target_pcd.get_axis_aligned_bounding_box()
        
        # Center both point clouds at origin
        ref_center = ref_bbox.get_center()
        target_center = target_bbox.get_center()
        
        ref_pcd_norm.translate(-ref_center)
        target_pcd_norm.translate(-target_center)
        
        print(f"Centered point clouds (translation applied)")
        
        # Handle extreme scale differences
        if scale_ratio > 10 or scale_ratio < 0.1:
            print(f"Extreme scale difference detected (ratio: {scale_ratio:.2f}), applying scale normalization")
            
            # Scale reference to match target approximately
            if scale_ratio > 10:
                # Reference is much larger, scale it down
                scale_factor = 1.0 / scale_ratio
                ref_pcd_norm.scale(scale_factor, center=(0, 0, 0))
                print(f"Scaled reference down by factor {scale_factor:.4f}")
            elif scale_ratio < 0.1:
                # Reference is much smaller, scale it up
                scale_factor = 1.0 / scale_ratio
                ref_pcd_norm.scale(scale_factor, center=(0, 0, 0))
                print(f"Scaled reference up by factor {scale_factor:.4f}")
        
        # Handle large center distances
        if center_distance > 100:
            print(f"Large center distance ({center_distance:.2f}), centering applied")
        
        # Create transformation record
        scale_transform = {
            'ref_translation': -ref_center,
            'target_translation': -target_center,
            'scale_factor': 1.0/scale_ratio if (scale_ratio > 10 or scale_ratio < 0.1) else 1.0,
            'applied_scaling': (scale_ratio > 10 or scale_ratio < 0.1)
        }
        
        return ref_pcd_norm, target_pcd_norm, scale_transform
    
    def _preprocess_point_clouds(self, ref_pcd, target_pcd, voxel_size):
        '''
        Preprocess point clouds: downsample, estimate normals, remove outliers
        '''
        print("Preprocessing point clouds...")
        
        # Downsample
        ref_down = ref_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        # Estimate normals
        ref_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # Remove statistical outliers
        ref_clean, _ = ref_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        target_clean, _ = target_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        print(f"After preprocessing: \nReference: {len(ref_clean.points)} points, Target: {len(target_clean.points)} points")
        
        return ref_clean, target_clean
    
    def _ppf_matching(self, ref_pcd, target_pcd, voxel_size):
        '''
        Perform Point Pair Features (PPF) matching for initial alignment
        '''
        try:
            # Prepare PPF matching
            distance_threshold = voxel_size * 1.5
            
            # Use RANSAC-based global registration with FPFH features
            # Compute FPFH features
            radius_feature = voxel_size * 5
            ref_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                ref_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            # RANSAC-based global registration
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                ref_pcd, target_pcd, ref_fpfh, target_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            
            if result.fitness > 1:  # Reasonable fitness threshold
                print(f"PPF/RANSAC success - Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
                return result.transformation
            else:
                print(f"PPF/RANSAC poor result - Fitness: {result.fitness:.4f}")
                return None
                
        except Exception as e:
            print(f"PPF matching failed: {e}")
            return None
    
    def _icp_refinement(self, ref_pcd, target_pcd, distance_threshold, 
                       relative_fitness, relative_rmse, max_iteration):
        '''
        Perform ICP refinement for precise alignment
        '''
        # Point-to-point ICP
        result_p2p = o3d.pipelines.registration.registration_icp(
            ref_pcd, target_pcd, distance_threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=relative_fitness,
                relative_rmse=relative_rmse,
                max_iteration=max_iteration))
        
        print(f"Point-to-Point ICP - Fitness: {result_p2p.fitness:.4f}, RMSE: {result_p2p.inlier_rmse:.4f}")
        
        # Point-to-plane ICP (more accurate if normals are good)
        try:
            result_p2plane = o3d.pipelines.registration.registration_icp(
                ref_pcd, target_pcd, distance_threshold, result_p2p.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=relative_fitness,
                    relative_rmse=relative_rmse,
                    max_iteration=max_iteration))
            
            print(f"Point-to-Plane ICP - Fitness: {result_p2plane.fitness:.4f}, RMSE: {result_p2plane.inlier_rmse:.4f}")
            
            # Use better result
            if result_p2plane.fitness > result_p2p.fitness:
                return result_p2plane.transformation, result_p2plane.fitness, result_p2plane.inlier_rmse
        
        except Exception as e:
            print(f"Point-to-plane ICP failed, using point-to-point: {e}")
        
        return result_p2p.transformation, result_p2p.fitness, result_p2p.inlier_rmse
    
    def _calculate_alignment_metrics(self, ref_pcd, target_pcd, distance_threshold):
        '''
        Calculate alignment quality metrics
        '''
        # Build KD-tree for target
        target_tree = o3d.geometry.KDTreeFlann(target_pcd)
        
        distances = []
        for point in ref_pcd.points:
            [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
            distances.append(np.sqrt(dist[0]))
        
        distances = np.array(distances)
        
        metrics = {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances),
            'inlier_ratio': np.sum(distances < distance_threshold) / len(distances),
            'rmse': np.sqrt(np.mean(distances**2))
        }
        
        print("Alignment Metrics:")
        print(f"  Mean distance: {metrics['mean_distance']:.4f}")
        print(f"  Median distance: {metrics['median_distance']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Inlier ratio: {metrics['inlier_ratio']:.4f}")
        
        return metrics
    
    def _save_matching_results(self, results, store_path):
        '''
        Save matching results to files
        '''
        results_dir = os.path.join(self._sparse_path, store_path, "surface_matching")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save aligned reference point cloud
        aligned_path = os.path.join(results_dir, "reference_aligned.ply")
        o3d.io.write_point_cloud(aligned_path, results['ref_pcd_aligned'])
        
        # Save transformation matrices
        transforms_path = os.path.join(results_dir, "transformations.txt")
        with open(transforms_path, 'w') as f:
            f.write("Initial Transformation (PPF):\n")
            np.savetxt(f, results['initial_transform'], fmt='%.6f')
            f.write("\nFinal Transformation (ICP):\n")
            np.savetxt(f, results['final_transform'], fmt='%.6f')
            f.write("\nCombined Transformation:\n")
            np.savetxt(f, results['combined_transform'], fmt='%.6f')
        
        # Save metrics
        metrics_path = os.path.join(results_dir, "alignment_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"ICP Fitness: {results['icp_fitness']:.6f}\n")
            f.write(f"ICP RMSE: {results['icp_rmse']:.6f}\n")
            for key, value in results['alignment_metrics'].items():
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"Results saved to: {results_dir}")
    
    def _visualize_matching_results(self, results):
        '''
        Visualize the surface matching results
        '''
        print("Visualizing surface matching results...")
        
        # Color point clouds differently
        ref_aligned = results['ref_pcd_aligned']
        target = results['target_pcd']
        
        # Paint reference red, target blue
        ref_aligned.paint_uniform_color([1, 0, 0])  # Red
        target.paint_uniform_color([0, 0, 1])       # Blue
        
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Visualize
        try:
            o3d.visualization.draw_geometries([ref_aligned, target, coord_frame],
                                            window_name="Surface Matching Results",
                                            width=1200, height=800)
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Results saved to files for external viewing")

        return
    
    def run_surface_matching_pipeline(self, store_path="0"):
        '''
        Run complete surface matching pipeline with multiple parameter sets
        to find best alignment between reference model and reconstructed point cloud.
        '''
        print("=== Running Complete Surface Matching Pipeline ===")
        
        # Parameter sets to try (coarse to fine)
        parameter_sets = [
            {
                'name': 'coarse',
                'voxel_size': 0.1,
                'distance_threshold': 0.05,
                'max_iteration': 1000
            },
            {
                'name': 'medium', 
                'voxel_size': 0.05,
                'distance_threshold': 0.02,
                'max_iteration': 2000
            },
            {
                'name': 'fine',
                'voxel_size': 0.02,
                'distance_threshold': 0.01,
                'max_iteration': 3000
            }
        ]
        
        best_result = None
        best_fitness = 0
        
        for i, params in enumerate(parameter_sets):
            print(f"\n--- Running {params['name']} alignment (Set {i+1}/3) ---")
            
            result = self.surface_matching_ppf_icp(
                store_path=store_path,
                voxel_size=params['voxel_size'],
                distance_threshold=params['distance_threshold'],
                max_iteration=params['max_iteration']
            )
            
            if result and result['icp_fitness'] > best_fitness:
                best_fitness = result['icp_fitness']
                best_result = result
                best_result['parameter_set'] = params['name']
        
        if best_result:
            print(f"\n=== Best Result: {best_result['parameter_set']} alignment ===")
            print(f"Final Fitness: {best_result['icp_fitness']:.4f}")
            print(f"Final RMSE: {best_result['icp_rmse']:.4f}")
            
            # Save best result summary
            summary_path = os.path.join(self._sparse_path, store_path, "surface_matching", "best_result_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Best Surface Matching Result\n")
                f.write(f"Parameter Set: {best_result['parameter_set']}\n")
                f.write(f"ICP Fitness: {best_result['icp_fitness']:.6f}\n")
                f.write(f"ICP RMSE: {best_result['icp_rmse']:.6f}\n")
                f.write(f"Mean Distance: {best_result['alignment_metrics']['mean_distance']:.6f}\n")
                f.write(f"Inlier Ratio: {best_result['alignment_metrics']['inlier_ratio']:.6f}\n")
        else:
            print("No successful surface matching results obtained.")
        
        return best_result
    
    def diagnose_alignment_failure(self, store_path="0"):
        '''
        Diagnose why surface matching failed by analyzing point cloud properties.
        '''
        print("=== ALIGNMENT FAILURE DIAGNOSIS ===")
        
        # Load point clouds
        ref_path = os.path.join(self._sat_model_path, "reference.ply")
        
        # Try multiple possible locations for target point cloud
        possible_paths = [
            os.path.join(self._sat_model_path, "points.ply"),
            os.path.join(store_path, "points.ply"),
            os.path.join(self._sparse_path, store_path, "points.ply"),
        ]
        
        if not os.path.exists(ref_path):
            print(f"Reference model not found: {ref_path}")
            return
        
        pcd_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pcd_path = path
                print(f"Found target point cloud at: {pcd_path}")
                break
        
        if pcd_path is None:
            print(f"Target point cloud not found in any location:")
            for path in possible_paths:
                print(f"    {path}")
            return
        
        ref_mesh = o3d.io.read_triangle_mesh(ref_path)
        ref_pcd = ref_mesh.sample_points_uniformly(number_of_points=10000)
        target_pcd = o3d.io.read_point_cloud(pcd_path)
        
        print(f"Loaded reference: {len(ref_pcd.points)} points")
        print(f"Loaded target: {len(target_pcd.points)} points")
        
        # Analyze properties
        ref_bbox = ref_pcd.get_axis_aligned_bounding_box()
        target_bbox = target_pcd.get_axis_aligned_bounding_box()
        
        ref_center = ref_bbox.get_center()
        target_center = target_bbox.get_center()
        ref_extent = ref_bbox.get_extent()
        target_extent = target_bbox.get_extent()
        
        center_distance = np.linalg.norm(ref_center - target_center)
        scale_ratio = np.max(ref_extent) / np.max(target_extent)
        
        print(f"\nSPATIAL ANALYSIS:")
        print(f"Reference center: [{ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f}]")
        print(f"Target center:    [{target_center[0]:.2f}, {target_center[1]:.2f}, {target_center[2]:.2f}]")
        print(f"Center distance:  {center_distance:.2f}")
        
        print(f"\nSCALE ANALYSIS:")
        print(f"Reference extent: [{ref_extent[0]:.2f}, {ref_extent[1]:.2f}, {ref_extent[2]:.2f}]")
        print(f"Target extent:    [{target_extent[0]:.2f}, {target_extent[1]:.2f}, {target_extent[2]:.2f}]")
        print(f"Scale ratio (ref/target): {scale_ratio:.4f}")
        
        # Diagnosis
        print(f"\nDIAGNOSIS:")
        
        if center_distance > 100:
            print(f"Point clouds are very far apart (distance: {center_distance:.1f})")
        
        if scale_ratio > 10:
            print(f"Reference is much larger than target (ratio: {scale_ratio:.1f})")
        elif scale_ratio < 0.1:
            print(f"Reference is much smaller than target (ratio: {scale_ratio:.1f})")
        
        if len(target_pcd.points) < 100:
            print(f"Very few target points ({len(target_pcd.points)})")
        
        if np.max(target_extent) < 0.1:
            print(f"Target point cloud is extremely small (max extent: {np.max(target_extent):.4f})")
        
        
        return {
            'center_distance': center_distance,
            'scale_ratio': scale_ratio,
            'ref_points': len(ref_pcd.points),
            'target_points': len(target_pcd.points),
            'ref_extent': ref_extent,
            'target_extent': target_extent
        }
    
    ################### Internal Funcs #####################
    def _clean_up(self, paths): 
        
        # Clean each of the paths 
        for path in paths:
            if os.path.exists(path):
                print(f"Removing old file/folder: {path}")
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)

        return 

    def _make_clean_dirs( self, paths ): 
        for path in paths: 
            os.makedirs(path, exist_ok=True)

        return 

    ################### Plotting ##################### 
    def plot_reference_model(self, camera_scale=0.1): 
        '''
        Visualize the reference satellite model in 3D.
        '''
        print("=== Loading and visualizing reference satellite model ===")
        
        # Load reference model
        ref_path = os.path.join( self._sat_model_path, "reference.ply" )
        mesh = o3d.io.read_triangle_mesh( ref_path )
        
        # Create coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale)
        
        print("Visualizing reference model")
        try:
            # Try hardware-accelerated rendering first
            o3d.visualization.draw_geometries([mesh, coord_frame])
        except Exception as e:
            print(f"Hardware rendering failed: {e}")
            print("Attempting software rendering...")
            # Create visualizer with software rendering options
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            vis.add_geometry(coord_frame)
            vis.run()
            vis.destroy_window()
        
        return
       
    def plot_pointcloud(self, store_path="0", camera_scale=0.1):
            
        print("=== Loading and visualizing sparse point cloud with cameras ===")
        
        # Load point cloud 
        store_name = os.path.join( self._sparse_path, store_path ) 
        file_path =  os.path.join( self._sparse_path, store_path, "points.ply" )
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
        
        return    
    
    def plot_keypoints(self): 
        
        first_image_path = os.path.join(self._image_path, self._image_names[0])
        img = cv2.imread(first_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        img_kp = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title("First image SIFT keypoints (CPU)")
        plt.axis('off')
        plt.show()

        return 




