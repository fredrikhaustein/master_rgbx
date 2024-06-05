import os
from PIL import Image
import numpy as np

def compare_png_files(folder1, folder2):
    # List to store filenames with different counts of 1s and their differences
    different_files = []

    # Get the list of files in the folders
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    # Ensure both folders have the same files
    assert files1 == files2, "Folders do not contain the same files."

    # Iterate through the files and compare
    for filename in files1:
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)

        # Open the images and convert them to numpy arrays
        image1 = np.array(Image.open(file1_path))
        image2 = np.array(Image.open(file2_path))

        # Count the number of 1s in each image
        count1 = np.sum(image1 == 1)
        count2 = np.sum(image2 == 1)

        # Compare the counts and record the difference
        if count1 != count2:
            difference = abs(count1 - count2)
            different_files.append((filename, difference))

    # Sort the list by the difference in descending order
    different_files.sort(key=lambda x: x[1], reverse=True)

    # Print the list of files with different counts
    print("Files with different counts of 1s (sorted by difference):")
    for filename, difference in different_files:
        print(f"{filename}: {difference} difference")

# Example usage:
folder1 = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_manually_labling_ground_truth_png"
folder2 = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_ground_truth_png"

compare_png_files(folder1, folder2)
