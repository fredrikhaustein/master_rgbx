from PIL import Image
import numpy as np

def create_colored_image(binary_image_path, color_map):
    # Load the binary image
    image = Image.open(binary_image_path)
    image_data = np.array(image)

    # Apply the color map
    colored_image = np.array([color_map[value] for value in image_data.flatten()]).reshape(image_data.shape + (3,))
    
    # Convert numpy array back to PIL image
    return Image.fromarray(np.uint8(colored_image), 'RGB')

def overlay_images(aerial_path, ground_truth_path, prediction_path):
    # Define the color map
    color_map = {
        0: (0, 255, 0),  # Green for class 0
        1: (255, 0, 0),  # Red for class 1
    }
    
    # Create colored images for ground truth and prediction
    colored_ground_truth = create_colored_image(ground_truth_path, color_map)
    colored_prediction = create_colored_image(prediction_path, color_map)

    # Open the aerial image
    aerial = Image.open(aerial_path).convert("RGBA")

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


    # Save the composite images
    # aerial_with_ground_truth.save('/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/17_51_ground_truth_direct.png', 'PNG')
    aerial_with_prediction.save('/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/17_51_prediction_indirect.png', 'PNG')

    print("Images saved successfully.")


import os

def find_similar_filenames(folder1, folder2):
    # Get the list of filenames in each folder
    filenames1 = set(os.listdir(folder1))
    filenames2 = set(os.listdir(folder2))

    # Find the intersection of both sets to get the common filenames
    similar_filenames = filenames1.intersection(filenames2)

    return similar_filenames


# Example usage
# overlay_images('/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png/17_51.png', '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/merged_ground_truths_binary_building_png/17_51.png', '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_augmented_buildings_color/17_51.png')
overlay_images('/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png/17_51.png', '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/merged_ground_truths_binary_building_png/17_51.png', '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_impervious_invert_vegitation_color/17_51.png')
# Example usage
# folder_a = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_impervious_invert_vegitation_color'
# folder_b = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_augmented_buildings_color'
# common_files = find_similar_filenames(folder_a, folder_b)s
# print("Common files:", common_files)
# Common files: {'25_03.png', '40_09.png', '23_13.png', '12_02.png', '42_23.png', '39_23.png', '28_10.png', '17_51.png', '10_25.png', '22_29.png', '21_42.png', '07_38.png', '18_25.png', '23_31.png', '04_25.png', '30_15.png', '01_35.png'}