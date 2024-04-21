import os
import shutil
import numpy as np

def split_dataset_and_create_structure(dataset_name, rgb_folder_path, gt_folder_path, ndvi_folder_path, base_folder='datasets'):
    """
    Create a dataset folder structure, split the dataset into training, validation, and testing sets,
    and populate it with images from the specified folders.

    Parameters:
    - dataset_name: Name of the dataset.
    - rgb_folder_path: Folder path containing RGB images.
    - gt_folder_path: Folder path containing ground truth images.
    - ndvi_folder_path: Folder path containing NDVI images.
    - base_folder: Base folder where the dataset structure will be created.
    """
    dataset_path = os.path.join(base_folder, dataset_name)
    rgb_folder = os.path.join(dataset_path, 'RGBFolder')
    modal_x_folder = os.path.join(dataset_path, 'ModalXFolder')
    label_folder = os.path.join(dataset_path, 'LabelFolder')
    
    # Create directories
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(modal_x_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    
    all_items = [os.path.splitext(filename)[0] for filename in os.listdir(rgb_folder_path)]
    np.random.shuffle(all_items)  # Randomly shuffle the items
    
    # Split the dataset
    total_items = len(all_items)
    train_end = int(total_items * 0.8)
    val_end = train_end + int(total_items * 0.1)
    
    train_items = all_items[:train_end]
    val_items = all_items[train_end:val_end]
    test_items = all_items[val_end:]

    def copy_files(items, src_folder, dst_folder, extension):
        for name_base in items:
            src_file = os.path.join(src_folder, f'{name_base}{extension}')
            dst_file = os.path.join(dst_folder, f'{name_base}{extension}')
            shutil.copy(src_file, dst_file)
    
    # Copy files to their respective directories
    copy_files(all_items, rgb_folder_path, rgb_folder, '.png')  # Adjust if RGB format differs
    copy_files(all_items, ndvi_folder_path, modal_x_folder, '.tif')  # Adjust if NDVI format differs
    copy_files(all_items, gt_folder_path, label_folder, '.png')  # Adjust if GT format differs
    
    # Write the train.txt, val.txt, and test.txt files
    for split_name, items in zip(['train.txt', 'val.txt', 'test.txt'], [train_items, val_items, test_items]):
        with open(os.path.join(dataset_path, split_name), 'w') as f:
            for item in items:
                f.write(f'{item}\n')

import os

def delete_selected_images(directory):
    """
    Deletes all images in the specified directory that end with '_flipped.tif' or '_rotated.tif'.

    Args:
    directory (str): The path to the directory where the files are located.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist:", directory)
        return

    # List all files in the directory
    files = os.listdir(directory)
    
    # Define suffixes to check for
    suffixes = ('_flipped.tif', '_rotated.tif')

    # Loop through files and delete those that end with the specified suffixes
    for file in files:
        if file.endswith(suffixes):
            os.remove(os.path.join(directory, file))
            print(f"Deleted: {file}")


# Example usage
rgb_folder_path = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png' 
# gt_folder_path_multi = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_merged_groundtruth_png_multiclass'
gt_folder_path_binary = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/merged_ground_truths_binary_building_png"
ndvi_folder_path = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi'
# dem_folder_path = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_dem_png"
dataset_name = 'Dataset_15_building_binary'

split_dataset_and_create_structure(dataset_name, rgb_folder_path, gt_folder_path_binary, ndvi_folder_path)
# delete_selected_images("/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi")