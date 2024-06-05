import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_image(path):
    """Load an image from a file path."""
    return np.array(Image.open(path))

def compare_images(image1, image2):
    """Compare two images and return the mask of differences."""
    if image1.shape != image2.shape:
        raise ValueError("Images do not have the same dimensions.")
    return image1 != image2

def save_difference(image1, image2, difference, save_path):
    """Save the image highlighting differences."""
    if len(image2.shape) == 2:  # Image is grayscale
        image2_color = np.stack([image2] * 3, axis=-1)  # Convert to RGB by duplicating the grayscale channel
    else:
        image2_color = image2
    
    # Create an image to highlight differences in red
    diff_image = np.where(difference[:,:,None], [255, 0, 0], image2_color)
    
    # Convert array back to an image
    diff_image = Image.fromarray(diff_image.astype('uint8'))
    diff_image.save(save_path)

def highlight_differences(folder1, folder2, output_folder):
    """Process all images in two folders, compare them, and save the differences highlighted in a new folder."""
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List images in both directories
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Loop over files
    for file in files1:
        if file in files2:
            path1 = os.path.join(folder1, file)
            path2 = os.path.join(folder2, file)
            save_path = os.path.join(output_folder, file)

            # Process each image pair
            image1 = load_image(path1)
            image2 = load_image(path2)

            # Find differences
            difference = compare_images(image1, image2)

            # Save the differences
            save_difference(image1, image2, difference, save_path)

# Define folders and output directory
pred_direct_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results_new_dataset/Dataset_17_fkb_mlpdecoder_indirect_ndvi_labeling'
ground_truth_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_17_fkb_impervious_surfaces/LabelFolder'
output_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/result_difference_ground_truth_pred/Dataset_17_fkb_mlpdecoder_indirect_ndvi_labeling'

# Call the function with folder paths
highlight_differences(pred_direct_folder, ground_truth_folder, output_folder)
