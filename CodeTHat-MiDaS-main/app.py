import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from point_cloud_utils import save_point_cloud, calculate_volume

# Load MiDaS model
print("Loading MiDaS model...")
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Load the field image
print("Loading field image...")
field_image = cv2.imread('field_image.jpg')
if field_image is None:
    raise FileNotFoundError("Field image 'field_image.jpg' not found!")

# Convert image to RGB
img_rgb = cv2.cvtColor(field_image, cv2.COLOR_BGR2RGB)

# Transform image for MiDaS
imgbatch = transform(img_rgb).to('cpu')

# Generate depth map
print("Generating depth map...")
with torch.no_grad():
    prediction = midas(imgbatch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode='bicubic',
        align_corners=False
    ).squeeze()

depth_map = prediction.cpu().numpy()

# Normalize depth map for visualization
normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# Visualize depth map
plt.figure(figsize=(10, 6))
plt.imshow(normalized_depth, cmap='inferno')
plt.colorbar(label="Depth")
plt.title("Normalized Depth Map")
plt.show()

# Save the depth map as a point cloud
print("Saving point cloud...")
save_point_cloud(depth_map, "field_point_cloud.ply")

# Calculate volume from the point cloud
print("Calculating volume...")
try:
    volume = calculate_volume("field_point_cloud.ply")
    print(f"Estimated Volume: {volume:.2f} cubic units")
except Exception as e:
    print(f"Error in volume calculation: {e}")
    volume = None

# Assume density and calculate mass if volume is valid
if volume is not None:
    density = 0.4  # Example density for weeds in kg/m^3
    mass = volume * density
    print(f"Estimated Mass: {mass:.2f} kg")
else:
    print("Mass calculation skipped due to invalid volume.")