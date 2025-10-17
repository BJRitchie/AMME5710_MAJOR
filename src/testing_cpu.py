import pycolmap
import os
import open3d as o3d


# DATASET_PATH = "/absolute/path/to/your/dataset"  # adjust this
# C:\Users\benri\OneDrive\Documents\University\4th Year\Sem 2\AMME5710\AMME5710_MAJOR\AMME5710_MAJOR\sparse\0
# rec_path = os.path.join(DATASET_PATH, "sparse/0")
rec_path = "sparse/0"

rec = pycolmap.Reconstruction(rec_path)
print("Registered images:", len(rec.images))
print("3D points:", len(rec.points3D))

# To save point cloud 
# rec = pycolmap.Reconstruction(rec_path)
# rec.export_PLY(os.path.join(rec_path, "points.ply"))
# print("Saved sparse point cloud to", os.path.join(rec_path, "points.ply"))


################### VISUALISATION #####################
# Visualize sparse reconstruction
# sparse_ply = os.path.join(inc_map_out_path, "0", "points.ply")
sparse_ply  = "sparse/0/points.ply"
if os.path.exists(sparse_ply):
    print("=== Loading and visualizing sparse point cloud ===")
    pcd = o3d.io.read_point_cloud(sparse_ply)
    o3d.visualization.draw_geometries([pcd])
else:
    print(f"No sparse point cloud found at {sparse_ply}. Sparse mapping might have failed.")



