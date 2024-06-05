from PIL import Image
import numpy as np
import os

def count_classes_in_images(folder_path):
    # Dictionary to store the count of each class across all images
    total_counts = {i: 0 for i in range(8)}  # Assuming classes 0 to 7
    total_pixels = 0  # Total number of pixels processed
    
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            
            # Open the image and convert it to a NumPy array
            with Image.open(file_path) as img:
                img_array = np.array(img)
                
                # Count occurrences of each class in the current image
                for i in range(8):
                    counts = np.sum(img_array == i)
                    total_counts[i] += counts
                    total_pixels += counts
    
    # Calculate and print the percentage of each class
    for class_id, count in total_counts.items():
        percent = (count / total_pixels) * 100
        print(f"Class {class_id}: {count} pixels, {percent:.2f}% of total")
    
    return total_counts

# Path to the folder containing the PNG files
folder_path = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_ground_truth_png'
folder_path2 = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_manually_labling_ground_truth_png"
folder_path3 = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/vegetation_ground_truth_ndvi_01_png"
count_classes_in_images(folder_path3)