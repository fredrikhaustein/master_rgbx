import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def highlight_differences(path1, path2, save_path):
    """Load two images, compare them, and save the differences highlighted."""
    # Load the images
    image1 = load_image(path1)
    image2 = load_image(path2)

    # Find differences
    difference = compare_images(image1, image2)

    # Save the differences
    save_difference(image1, image2, difference, save_path)

# Example usage:
pred_indirect = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_impervious_invert_vegitation/01_35.png'

pred_direct = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_augmented_building_color/01_35.png'

ground_truth = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_15_building_binary/LabelFolder/01_35.png'

save_path = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/01_35_difference_direct.png'

highlight_differences(pred_direct, ground_truth, save_path)
