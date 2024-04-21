import os
import cv2
import numpy as np

# Define your dataset folders
dataset_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_2'
rgb_folder = os.path.join(dataset_folder, 'RGBFolder')
modalx_folder = os.path.join(dataset_folder, 'ModalXFolder')
label_folder = os.path.join(dataset_folder, 'LabelFolder')
train_file = os.path.join(dataset_folder, 'train.txt')

# Load the list of training images
with open(train_file, 'r') as file:
    train_images = file.read().splitlines()

# Track images to remove due to having only label 0
images_to_remove = []

# Check each label image for labels other than 0
for image_name in train_images:
    label_image_path = os.path.join(label_folder, image_name + '.png')
    label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the label image contains only 0
    if np.all(label_image == 0):
        images_to_remove.append(image_name)

# Remove the identified images and their entries
for image_name in images_to_remove:
    os.remove(os.path.join(rgb_folder, image_name + '.png'))
    os.remove(os.path.join(modalx_folder, image_name + '.png'))
    os.remove(os.path.join(label_folder, image_name + '.png'))
    train_images.remove(image_name)

# Rewrite the train.txt file without the removed images
with open(train_file, 'w') as file:
    for image_name in train_images:
        file.write(image_name + '\n')

print(f"Removed {len(images_to_remove)} images that only had label 0.")
