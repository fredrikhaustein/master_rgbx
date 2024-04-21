import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2

# Define your dataset folders
dataset_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_2'
rgb_folder = os.path.join(dataset_folder, 'RGBFolder')
modalx_folder = os.path.join(dataset_folder, 'ModalXFolder')
label_folder = os.path.join(dataset_folder, 'LabelFolder')
train_file = os.path.join(dataset_folder, 'train.txt')

# Read your train.txt file to get the list of training images
with open(train_file, 'r') as file:
    train_images = file.readlines()

# Identify the images with label 2
images_to_augment = []
for image_name in train_images:
    label_image_path = os.path.join(label_folder, image_name.strip() + '.png')
    label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
    if 2 in label_image:  # Assuming label 2 is represented by the pixel value 2
        images_to_augment.append(image_name.strip())

# Define your augmentation pipeline
aug = A.Compose([
    A.Flip(p=0.5),  # Randomly flip the image either horizontally, vertically or both 
    A.RandomRotate90(p=0.5),  # Randomly rotate the image by 90 degrees zero or more times
    A.GaussianBlur(p=0.3, blur_limit=(3, 7)),  # Apply gaussian blur with a probability of 0.3
    # more augmentations can be added here
])

# Apply augmentations and save the new images
for base_name in images_to_augment:
    # Read the original images
    rgb_img = cv2.imread(os.path.join(rgb_folder, base_name + '.png'))
    modalx_img = cv2.imread(os.path.join(modalx_folder, base_name + '.png'))
    label_img = cv2.imread(os.path.join(label_folder, base_name + '.png'), cv2.IMREAD_GRAYSCALE)

    # Ensure images are in the same size, if not, resize modalx and label images to match rgb image
    # Note: This step is crucial to ensure consistent augmentation across the inputs
    modalx_img = cv2.resize(modalx_img, (rgb_img.shape[1], rgb_img.shape[0]))
    label_img = cv2.resize(label_img, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply the augmentations
    augmented = aug(image=rgb_img, image0=modalx_img, mask=label_img)
    rgb_img_aug, modalx_img_aug, label_img_aug = augmented["image"], augmented["image0"], augmented["mask"]

    # Save the augmented images with a new name to indicate they are augmented
    new_base_name = base_name + '_aug'
    cv2.imwrite(os.path.join(rgb_folder, new_base_name + '.png'), rgb_img_aug)
    cv2.imwrite(os.path.join(modalx_folder, new_base_name + '.png'), modalx_img_aug)
    cv2.imwrite(os.path.join(label_folder, new_base_name + '.png'), label_img_aug)

    # Update the training list with the new images
    with open(train_file, 'a') as file:
        file.write(new_base_name + '\n')
