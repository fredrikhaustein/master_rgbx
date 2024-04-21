from PIL import Image
import numpy as np
import os

def find_unique_pixel_values(folder_path):
    unique_values = set()  # Use a set to store unique values and avoid duplicates

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # Check if the file is a PNG image
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                img_array = np.array(img)  # Convert the image to a NumPy array
                unique_values_in_image = np.unique(img_array)  # Find unique values in this image
                unique_values.update(unique_values_in_image)  # Add these values to the set

    return unique_values

def find_unique_pixel_values_image(image_path):
    unique_values = set()  # Use a set to store unique values and avoid duplicates

    # Check if the file is a PNG image
    if image_path.endswith('.png'):
        with Image.open(image_path) as img:
            img_array = np.array(img)  # Convert the image to a NumPy array
            unique_values_in_image = np.unique(img_array)  # Find unique values in this image
            unique_values.update(unique_values_in_image)  # Add these values to the set

    return unique_values

folder_path = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_2/LabelFolder'
path1 ="/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/results_multiclass_ndvi_shadow_refined/09_03.png"
path2 = "/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/results_multiclass_ndvi_shadow/09_03.png"
unique_pixel_values = find_unique_pixel_values_image(path1)
unique_pixel_values2 = find_unique_pixel_values_image(path2)
print(unique_pixel_values)
print(unique_pixel_values2)