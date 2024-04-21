import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
import os

def augment_image_pil(img):
    """Apply augmentation to a PIL image: flipping and 90-degree rotation."""
    return [img.transpose(Image.FLIP_LEFT_RIGHT), img.rotate(90)]

def augment_image_rasterio(image_path, augmentation_type):
    """Augment a raster (.tif) image: flipping and 90-degree rotation."""
    with rasterio.open(image_path) as src:
        data = src.read()
        kwargs = src.meta
        
        if augmentation_type == "flipped":
            # Flip image horizontally
            data = data[:, :, ::-1]
        elif augmentation_type == "rotated":
            # Rotate image 90 degrees
            data = np.rot90(data, k=1, axes=(1, 2))

        return data, kwargs

def save_raster(output_path, data, kwargs):
    """Save augmented raster data to a .tif file."""
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write(data)

def should_augment(img_array, class_threshold=0.3, target_class=1):
    """Check if the target class occupies more than the specified threshold."""
    total_pixels = img_array.size
    target_pixels = np.sum(img_array == target_class)
    percent = (target_pixels / total_pixels)
    return percent > class_threshold

def process_and_augment_images(folder_ground_truth, folder_aerial, folder_ndvi,
                               output_folder_ground_truth, output_folder_aerial, output_folder_ndvi,
                               class_threshold=0.3, target_class=1):
    """Process and augment images based on the class threshold."""
    for path in [output_folder_ground_truth, output_folder_aerial, output_folder_ndvi]:
        os.makedirs(path, exist_ok=True)
        
    for filename in os.listdir(folder_ground_truth):
        if filename.endswith(".png"):  # Adjust if your ground truth images have a different format
            paths = {
                "gt": os.path.join(folder_ground_truth, filename),
                "aerial": os.path.join(folder_aerial, filename.replace('.png', '.tif')),  # Assuming aerial images are .tif
                "ndvi": os.path.join(folder_ndvi, filename.replace('.png', '.tif'))  # Adjust filename as needed
            }
            
            with Image.open(paths["gt"]) as gt_img:
                gt_array = np.array(gt_img)
                if should_augment(gt_array, class_threshold, target_class):
                    # Augmentations for ground truth and aerial (PIL Images)
                    augmented_images = {"gt": augment_image_pil(gt_img)}
                    
                    # Handling aerial images similarly to ground truth, adjust if needed
                    with Image.open(paths["aerial"].replace('.tif', '.png')) as aerial_img:  # Assuming aerial images are .png for simplicity
                        augmented_images["aerial"] = augment_image_pil(aerial_img)
                    
                    # Save original and augmented images
                    for img_type, imgs in augmented_images.items():
                        base_output_path = os.path.join(eval(f"output_folder_{img_type}"), os.path.splitext(filename)[0])
                        gt_img.save(f"{base_output_path}.png")
                        imgs[0].save(f"{base_output_path}_flipped.png")
                        imgs[1].save(f"{base_output_path}_rotated.png")
                    
                    # Augmentation for NDVI (Rasterio)
                    for aug_type in ["flipped", "rotated"]:
                        data, kwargs = augment_image_rasterio(paths["ndvi"], aug_type)
                        save_raster(f"{os.path.splitext(paths['ndvi'])[0]}_{aug_type}.tif", data, kwargs)
                else:
                    # Copy original images without augmentation if not exceeding threshold
                    gt_img.save(os.path.join(output_folder_ground_truth, filename))
                    Image.open(paths["aerial"].replace('.tif', '.png')).save(os.path.join(output_folder_aerial, filename))  # Assuming aerial images need conversion
                    # For NDVI, simply copy the file as is, assuming it's already in the correct format (.tif)
                    with open(paths["ndvi"], 'rb') as src_ndvi, open(os.path.join(output_folder_ndvi, filename.replace('.png', '.tif')), 'wb') as dst_ndvi:
                        dst_ndvi.write(src_ndvi.read())

# Update paths to your directories
folder_ground_truth = 'path_to_ground_truth_folder'
folder_aerial = 'path_to_aerial_images_folder'
folder_ndvi = 'path_to_ndvi_folder'

output_folder_ground_truth = 'path_to_output_ground_truth_folder'
output_folder_aerial = 'path_to_output_aerial_folder'
output_folder_ndvi = 'path_to_output_ndvi_folder'

process_and_augment_images(folder_ground_truth, folder_aerial, folder_ndvi,
                           output_folder_ground_truth, output_folder_aerial, output_folder_ndvi)