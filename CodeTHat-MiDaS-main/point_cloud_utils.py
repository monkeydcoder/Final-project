import numpy as np
import open3d as o3d

# Adjust camera intrinsic parameters based on the new image size (1440x959)
fx, fy = 1440, 959  # Focal lengths based on new image dimensions
cx, cy = 720, 479   # Principal point offsets (image center for 1440x959)

def depth_to_point_cloud(depth_map, img_width, img_height):
    """
    Convert a depth map to a 3D point cloud.

    Parameters:
    depth_map: 2D numpy array representing depth values.
    img_width: Width of the original image.
    img_height: Height of the original image.

    Returns:
    numpy array of shape (N, 3) representing 3D points.
    """
    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(img_width), np.arange(img_height))
    z = depth_map.flatten()

    # Convert pixel coordinates to normalized device coordinates
    x_normalized = (x.flatten() - cx) / fx
    y_normalized = (y.flatten() - cy) / fy

    # Convert to 3D points
    x_3d = x_normalized * z
    y_3d = y_normalized * z
    z_3d = z

    # Stack the coordinates into a (N, 3) array
    points_3d = np.vstack((x_3d, y_3d, z_3d)).T
    return points_3d

def save_point_cloud(depth_map, file_name):
    """
    Save the depth map as a 3D point cloud file.

    Parameters:
    depth_map: 2D numpy array representing depth values.
    file_name: Name of the output file.
    """
    # Use the image dimensions from the depth map
    img_width, img_height = depth_map.shape[1], depth_map.shape[0]
    points_3d = depth_to_point_cloud(depth_map, img_width, img_height)

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    # Save the point cloud to a .ply file
    o3d.io.write_point_cloud(file_name, point_cloud)
    print(f"3D Point Cloud saved as '{file_name}'")

def calculate_volume(file_name):
    """
    Calculate the volume of a 3D point cloud.

    Parameters:
    file_name: Path to the point cloud file.

    Returns:
    float: Estimated volume.
    """
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(file_name)

    # Estimate volume using a mesh reconstruction method
    # Adjust alpha value for better mesh fitting
    alpha = 0.05  # Larger alpha for complex shapes
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=alpha)

    # Ensure the mesh is watertight before estimating volume
    if not mesh.is_watertight():
        print("Warning: Mesh is not watertight. Volume estimate may be inaccurate.")

    volume = mesh.get_volume()
    return volume