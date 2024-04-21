import os
import shutil
import numpy as np
import rasterio

# Directories
ndvi_dir = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi"
aerial_dir = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png'
ground_truth_dir = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_merged_groundtruth_png_multiclass'

# New directories to create
new_ndvi_dir = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi_multiclass_filtered'
new_aerial_dir = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png_multiclass_filtered'
new_ground_truth_dir = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_merged_groundtruth_png_multiclass_filtered'

# Create new directories if they don't exist
for dir_path in [new_ndvi_dir, new_aerial_dir, new_ground_truth_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def calculate_ndvi_percentage(ndvi_image_path):
    with rasterio.open(ndvi_image_path) as src:
        ndvi = src.read(1)  # Assuming NDVI is the first band
        above_threshold = ndvi > -0.05
        percentage = np.mean(above_threshold) * 100  # Convert fraction to percentage
    return percentage

def copy_files(files, destination):
    for file in files:
        shutil.copy(file, destination)

for filename in os.listdir(ndvi_dir):
    ndvi_path = os.path.join(ndvi_dir, filename)
    percentage = calculate_ndvi_percentage(ndvi_path)
    if percentage > 50:
        # Assuming NDVI files have extensions that need to be replaced with .png for aerial and ground truth
        base_filename = os.path.splitext(filename)[0] + '.png'
        aerial_path = os.path.join(aerial_dir, base_filename)
        ground_truth_path = os.path.join(ground_truth_dir, base_filename)
        # Copy files to new directories
        copy_files([ndvi_path], new_ndvi_dir)
        if os.path.exists(aerial_path):  # Ensure the file exists before copying
            copy_files([aerial_path], new_aerial_dir)
        if os.path.exists(ground_truth_path):  # Ensure the file exists before copying
            copy_files([ground_truth_path], new_ground_truth_dir)

print("Processing complete.")
