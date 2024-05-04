import numpy as np
from PIL import Image
import os

def load_and_resize_image(path, desired_size):
    """Load an image from a file path and resize it to the desired size."""
    image = Image.open(path)
    if image.size != desired_size:
        image = image.resize(desired_size, Image.NEAREST)  # Resize using nearest neighbor to avoid interpolation issues with labels.
    return np.array(image)

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
    diff_image = np.where(difference[:, :, None], [255, 0, 0], image2_color)
    
    # Convert array back to an image
    diff_image = Image.fromarray(diff_image.astype('uint8'))
    diff_image.save(save_path)

def create_colored_image(binary_image_path, color_map, desired_size):
    # Load the binary image
    image = Image.open(binary_image_path)
    image_data = np.array(image)

    # Resize and pad the image if necessary
    if image_data.shape != desired_size:
        # Resize the image to the correct dimensions
        image = image.resize((desired_size[1], desired_size[0]), Image.NEAREST)  # Resize image to match the width and height
        image_data = np.array(image)

    # Apply the color map
    colored_image = np.array([color_map[value] for value in np.ravel(image_data)]).reshape(image_data.shape + (3,))
    
    # Convert numpy array back to PIL image
    return Image.fromarray(np.uint8(colored_image), 'RGB')

def overlay_images(aerial_path, ground_truth_path, prediction_path, output_path):
    # Define the color map
    color_map = {
        0: (0, 255, 0),  # Green for class 0
        1: (255, 0, 0),  # Red for class 1
    }
    
    # Open the aerial image and convert to RGBA
    aerial = Image.open(aerial_path).convert("RGBA")
    aerial_size = aerial.size

    # Create colored images for ground truth and prediction
    colored_ground_truth = create_colored_image(ground_truth_path, color_map, aerial_size)
    colored_prediction = create_colored_image(prediction_path, color_map, aerial_size)

    # Convert colored images to RGBA for alpha compositing
    colored_ground_truth = colored_ground_truth.convert("RGBA")
    colored_prediction = colored_prediction.convert("RGBA")

    # Set alpha for transparency in the overlay
    alpha = 128  # Adjust alpha between 0 (transparent) to 255 (opaque)
    colored_ground_truth.putalpha(alpha)
    colored_prediction.putalpha(alpha)

    # Composite the images
    aerial_with_ground_truth = Image.alpha_composite(aerial, colored_ground_truth)
    aerial_with_prediction = Image.alpha_composite(aerial, colored_prediction)

    # Save the composite image
    output_filename = os.path.join(output_path, 'overlay_' + os.path.basename(aerial_path))
    aerial_with_prediction.save(output_filename, 'PNG')
    print(f"Overlay image saved successfully at {output_filename}.")

def highlight_differences(path1, path2, output_folder):
    """Load two images, ensure they are the same size, compare them, and save the differences highlighted."""
    # Determine the desired size based on the first image.
    image1 = Image.open(path1)
    desired_size = image1.size

    # Load and resize both images
    image1 = load_and_resize_image(path1, desired_size)
    image2 = load_and_resize_image(path2, desired_size)

    # Compare the images
    difference = compare_images(image1, image2)

    # Save the differences
    save_path = os.path.join(output_folder, 'difference_' + os.path.basename(path1))
    save_difference(image1, image2, difference, save_path)
    print(f"Difference image saved successfully at {save_path}.")

def overlay_white_on_predictions(aerial_path, prediction_path, output_folder):
    """Overlay white color on aerial image where prediction is 1."""
    # Load the aerial image
    aerial = Image.open(aerial_path)
    aerial_data = np.array(aerial)

    # Load the prediction image
    prediction = Image.open(prediction_path)
    prediction = prediction.resize(aerial.size, Image.NEAREST)  # Ensure prediction is same size as aerial
    prediction_data = np.array(prediction)

    # Check if aerial image is in RGB format, convert if not
    if len(aerial_data.shape) < 3 or aerial_data.shape[2] != 3:
        aerial = aerial.convert('RGB')
        aerial_data = np.array(aerial)

    # Create a white overlay where prediction is 1
    white_overlay = np.where(prediction_data[..., None] == 1, [255, 255, 255], aerial_data)

    # Convert the numpy array back to an image
    output_image = Image.fromarray(white_overlay.astype('uint8'))

    # Save the output image
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, 'white_overlay_' + os.path.basename(aerial_path))
    output_image.save(output_filename, 'PNG')
    print(f"Overlay image with white highlights saved successfully at {output_filename}.")


def process_all_common_files(base_folder, predictions_folder, ground_truth_folder, output_folder):
    # Get common filenames
    prediction_files = set(os.listdir(predictions_folder))
    ground_truth_files = set(os.listdir(ground_truth_folder))
    common_files = prediction_files.intersection(ground_truth_files)

    # Create output folder if it does not exist
    output_path = output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process each file
    for file_name in common_files:
        aerial_path = os.path.join(base_folder, file_name)
        ground_truth_path = os.path.join(ground_truth_folder, file_name)
        prediction_path = os.path.join(predictions_folder, file_name)

        # Overlay images with white where prediction is 1
        overlay_white_on_predictions(aerial_path, prediction_path, output_path)
        # Overlay images
        # overlay_images(aerial_path, ground_truth_path, prediction_path, output_path)

        # # Highlight differences
        # highlight_differences(prediction_path, ground_truth_path, output_path)




# Example usage
# base_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_14_vegetation_ground_truth_binary/RGBFolder'
# predictions_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_augmented_building'
# ground_truth_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_15_building_binary/LabelFolder'
# output_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images_direct_method_overlay_white'


# Example usage Oslo
base_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_13_test_oslo/RGBFolder'
predictions_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/oslo_test_area'
ground_truth_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_13_test_oslo/LabelFolder'
output_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_oslo_master_images_direct_method_overlay_white'

process_all_common_files(base_folder, predictions_folder, ground_truth_folder, output_folder)