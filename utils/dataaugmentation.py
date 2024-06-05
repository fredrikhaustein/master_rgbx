from PIL import Image
import numpy as np
import rasterio
import os

def augment_image_pil(img):
    """Applies horizontal flip and 90-degree rotation to a PIL image."""
    return [img.transpose(Image.FLIP_LEFT_RIGHT), img.rotate(90)]

def augment_image_rasterio(image_path, augmentation_type):
    """Applies augmentation to a raster (.tif) image: horizontal flip or 90-degree rotation."""
    with rasterio.open(image_path) as src:
        kwargs = src.meta.copy()
        if augmentation_type == "flipped":
            # Horizontal flip. Note: rasterio reads in (bands, rows, cols)
            data = src.read()[:, :, ::-1]
        elif augmentation_type == "rotated":
            # 90-degree rotation
            data = np.rot90(src.read(), k=1, axes=(1, 2))
        return data, kwargs

def save_raster(output_path, data, kwargs):
    """Saves augmented raster data to a .tif file."""
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write(data)

def should_augment(img_array, class_threshold=0.4, target_class=1):
    """Checks if the target class occupies more than the specified threshold."""
    total_pixels = img_array.size
    target_pixels = np.sum(img_array == target_class)
    percent = (target_pixels / total_pixels)
    return percent > class_threshold

def process_and_augment_images(folder_ground_truth, folder_aerial, folder_ndvi,
                               output_folder_gt, output_folder_aerial, output_folder_ndvi,
                               class_threshold=0.4, target_class=1):
    """Process and augment images based on the class threshold."""
    for path in [output_folder_gt, output_folder_aerial, output_folder_ndvi]:
        os.makedirs(path, exist_ok=True)
    
    for filename in os.listdir(folder_ground_truth):
        if filename.endswith(".png"):
            gt_path = os.path.join(folder_ground_truth, filename)
            aerial_path = os.path.join(folder_aerial, filename)
            ndvi_path = os.path.join(folder_ndvi, filename.replace('.png', '.tif'))
            
            with Image.open(gt_path) as gt_img:
                gt_array = np.array(gt_img)
                if should_augment(gt_array, class_threshold, target_class):
                    # Perform and save augmentations for ground truth and aerial images
                    for img, folder in [(gt_img, output_folder_gt), (Image.open(aerial_path), output_folder_aerial)]:
                        for aug, suffix in zip(augment_image_pil(img), ["_flipped", "_rotated"]):
                            aug.save(os.path.join(folder, f"{os.path.splitext(filename)[0]}{suffix}.png"))
                    
                    # Perform and save augmentations for NDVI (.tif)
                    for aug_type in ["flipped", "rotated"]:
                        data, kwargs = augment_image_rasterio(ndvi_path, aug_type)
                        augmented_ndvi_path = os.path.join(output_folder_ndvi, f"{os.path.splitext(os.path.basename(ndvi_path))[0]}_{aug_type}.tif")
                        save_raster(augmented_ndvi_path, data, kwargs)

                # Copy original images to output folders
                gt_img.save(os.path.join(output_folder_gt, filename))
                Image.open(aerial_path).save(os.path.join(output_folder_aerial, filename))
                # NDVI requires handling as raster due to its format
                with rasterio.open(ndvi_path) as src_ndvi:
                    kwargs = src_ndvi.meta.copy()
                    data = src_ndvi.read()
                    save_raster(os.path.join(output_folder_ndvi, os.path.basename(ndvi_path)), data, kwargs)


# Update these paths to your directories
folder_ground_truth = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/vegetation_ground_truth_NDVI_png'
folder_aerial = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png'
folder_ndvi = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi'

output_folder_ground_truth = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_manually_labling_ground_truth_png_augmented_indirect'
output_folder_aerial = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image_png_augmented_indirect'
output_folder_ndvi = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/tiles_infraRed_shadow_ndvi_augmented_indirect'

process_and_augment_images(folder_ground_truth, folder_aerial, folder_ndvi,
                           output_folder_ground_truth, output_folder_aerial, output_folder_ndvi)
