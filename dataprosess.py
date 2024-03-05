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
    copy_files(all_items, ndvi_folder_path, modal_x_folder, '.png')  # Adjust if NDVI format differs
    copy_files(all_items, gt_folder_path, label_folder, '.png')  # Adjust if GT format differs
    
    # Write the train.txt, val.txt, and test.txt files
    for split_name, items in zip(['train.txt', 'val.txt', 'test.txt'], [train_items, val_items, test_items]):
        with open(os.path.join(dataset_path, split_name), 'w') as f:
            for item in items:
                f.write(f'{item}\n')

# Example usage
rgb_folder_path = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png' 
gt_folder_path = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_merged_groundtruth_png_multiclass'
ndvi_folder_path = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi_png'
dataset_name = 'Dataset_2'

split_dataset_and_create_structure(dataset_name, rgb_folder_path, gt_folder_path, ndvi_folder_path)
