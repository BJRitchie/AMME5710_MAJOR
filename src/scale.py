import numpy as np
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob 

def draw_hsv_box(lower_hsv, upper_hsv, axis='HV', color='black'):
    """
    Draw a rectangle for HSV thresholds on a scatter plot.
    lower_hsv, upper_hsv: [H, S, V] in [0,1]
    axis: 'HV' or 'HS'
    """
    if axis == 'HV':
        rect = plt.Rectangle(
            (lower_hsv[0], lower_hsv[2]),
            upper_hsv[0]-lower_hsv[0],
            upper_hsv[2]-lower_hsv[2],
            edgecolor=color, facecolor='none', linewidth=2
        )
    elif axis == 'HS':
        rect = plt.Rectangle(
            (lower_hsv[0], lower_hsv[1]),
            upper_hsv[0]-lower_hsv[0],
            upper_hsv[1]-lower_hsv[1],
            edgecolor=color, facecolor='none', linewidth=2
        )
    plt.gca().add_patch(rect)

def hsv_scatter_plot(points_rgb, thresholds=None):
    """
    Plot HSV distributions of a point cloud with optional threshold boxes.
    
    points_rgb: Nx3 array of RGB colors in [0,1]
    thresholds: list of (lower_hsv, upper_hsv) tuples to plot boxes
    """
    # Convert RGB to HSV
    hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in points_rgb])
    H, S, V = hsv_colors[:,0], hsv_colors[:,1], hsv_colors[:,2]

    # Keep original RGB for coloring
    RGB_colors = points_rgb

    # -----------------------------
    # Hue vs Value
    plt.figure(figsize=(6,5))
    plt.scatter(H, V, c=RGB_colors, s=5)
    plt.xlabel('Hue')
    plt.ylabel('Value')
    plt.title('Hue vs Value')
    if thresholds is not None:
        for lower, upper in thresholds:
            draw_hsv_box(lower, upper, axis='HV', color='black')
    plt.show()

    # -----------------------------
    # Hue vs Saturation
    plt.figure(figsize=(6,5))
    plt.scatter(H, S, c=RGB_colors, s=5)
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.title('Hue vs Saturation')
    if thresholds is not None:
        for lower, upper in thresholds:
            draw_hsv_box(lower, upper, axis='HS', color='black')
    plt.show()


if __name__ == "__main__":
    import open3d as o3d
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import colorsys

    # -----------------------------
    # Load point cloud
    # -----------------------------
    print("Load point cloud")
    pcd = o3d.io.read_point_cloud("sparse/0/points.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # RGB in [0,1]
    print(f"Loaded point cloud with {points.shape[0]} points")

    # -----------------------------
    # 2. Convert RGB to HSV
    # -----------------------------
    print("Convert RGB to HSV")
    hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in colors])
    
    # -----------------------------
    # Plot HSV Colour Space with Thresholds
    # -----------------------------
    points_rgb = np.asarray(pcd.colors)  # Nx3, RGB in [0,1]

    # Thresholds
    H_min, H_max = 0.1, 0.2
    S_min, S_max = 0.5, 1.0
    V_min, V_max = 0.2, 1.0
    thresholds = [
        ([H_min, S_min, V_min], [H_max, S_max, V_max])  
    ]

    hsv_scatter_plot(points_rgb, thresholds=thresholds)

    # -----------------------------
    # HSV thresholding
    # -----------------------------
    mask = ((hsv_colors[:,0] >= H_min) & (hsv_colors[:,0] <= H_max) &
            (hsv_colors[:,1] >= S_min) & (hsv_colors[:,1] <= S_max) &
            (hsv_colors[:,2] >= V_min) & (hsv_colors[:,2] <= V_max))
    
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    print(f"Filtered to {filtered_points.shape[0]} points based on HSV")

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Points by HSV")

    # -----------------------------
    # Cluster filtered points (k=2)
    # -----------------------------
    kmeans = KMeans(n_clusters=2, random_state=42).fit(filtered_points)
    labels = kmeans.labels_
    
    centroids = kmeans.cluster_centers_
    print(f"Centroids:\n{centroids}")
    
    cluster_pcds = []
    colors_palette = [[1,0,0], [0,1,0]]  # for visualization
    for i in range(2):
        cluster_points = filtered_points[labels==i]
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
        pcd_cluster.paint_uniform_color(colors_palette[i])
        cluster_pcds.append(pcd_cluster)

    # -----------------------------
    # Draw line between centroids
    # -----------------------------
    line_points = centroids
    lines = [[0,1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([0, 0, 0])  # black line

    o3d.visualization.draw_geometries(cluster_pcds + [line_set], window_name="Clusters with Centroid Line")

    # -----------------------------
    # Compute distance and scale factor
    # -----------------------------
    known_distance = 0.05  # meters, for example
    measured_distance = np.linalg.norm(centroids[0] - centroids[1])
    scale_factor = known_distance / measured_distance
    print(f"Measured distance: {measured_distance:.6f}, True distance: {known_distance}m → Scale factor: {scale_factor:.6f}")


    # -----------------------------
    # Apply scale factor to original point cloud
    # -----------------------------
    points_scaled = points * scale_factor
    scaled_pcd = o3d.geometry.PointCloud()
    scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)
    scaled_pcd.colors = o3d.utility.Vector3dVector(colors)  # keep original colors


    # -----------------------------
    # Assume scaled_pcd is your scaled Open3D point cloud
    # -----------------------------
    points = np.asarray(scaled_pcd.points)
    colors = np.asarray(scaled_pcd.colors)

    # -----------------------------
    # Draw Open3D point cloud
    # -----------------------------
    o3d.visualization.draw_geometries([scaled_pcd], window_name="Scaled Point Cloud")

    # -----------------------------
    # Matplotlib overlay for axes labels
    # -----------------------------
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter original points
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=1)

    # Draw axes lines (1 cm = 0.01 m)
    axis_length = 0.01
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=2, arrow_length_ratio=0.1)

    # Axis labels
    ax.text(axis_length*1.05, 0, 0, "X", color='r', fontsize=12)
    ax.text(0, axis_length*1.05, 0, "Y", color='g', fontsize=12)
    ax.text(0, 0, axis_length*1.05, "Z", color='b', fontsize=12)
    ax.text(axis_length*0.5, axis_length*0.5, axis_length*0.5, "1 cm", color='k', fontsize=10)

    # Set aspect ratio equal
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Scaled Point Cloud with 1 cm Axis Reference")
    plt.show()



    # ## one centimetre coordinate axes labels TODO Extend and add bubbles every centimetre? 
    # # TODO make a bounding box around satellite? 
    # # -----------------------------
    # # Create axis lines of 1 cm
    # # -----------------------------
    # axis_length = 0.01  # 1 cm
    # axis_points = np.array([
    #     [0, 0, 0],          # origin
    #     [axis_length, 0, 0], # X-axis
    #     [0, axis_length, 0], # Y-axis
    #     [0, 0, axis_length]  # Z-axis
    # ])

    # # Lines: origin to each axis tip
    # lines = [[0, 1], [0, 2], [0, 3]]
    # axis_line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(axis_points),
    #     lines=o3d.utility.Vector2iVector(lines)
    # )
    # axis_line_set.colors = o3d.utility.Vector3dVector([
    #     [1, 0, 0],  # X = red
    #     [0, 1, 0],  # Y = green
    #     [0, 0, 1]   # Z = blue
    # ])

    # # -----------------------------
    # # Visualize scaled point cloud with axes
    # # -----------------------------
    # o3d.visualization.draw_geometries([scaled_pcd, axis_line_set],
    #                                 window_name="Scaled Point Cloud with 1 cm Axes")




    # Original no axes labels
    # # -----------------------------
    # # Apply scale factor to the original point cloud
    # # -----------------------------
    # points_scaled = points * scale_factor
    # scaled_pcd = o3d.geometry.PointCloud()
    # scaled_pcd.points = o3d.utility.Vector3dVector(points_scaled)

    # # Preserve original colors
    # scaled_pcd.colors = o3d.utility.Vector3dVector(colors)

    # # Visualize scaled point cloud
    # o3d.visualization.draw_geometries([scaled_pcd],
    #                                 window_name="Scaled Original Point Cloud")


