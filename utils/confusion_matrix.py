import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_and_save_confusion_matrix_percentage(npy_file_path, output_file_path):
    """
    Load a confusion matrix from a .npy file, convert it to percentages, display and save it using matplotlib.

    Parameters:
    - npy_file_path: Path to the .npy file containing the confusion matrix.
    - output_file_path: Path to save the output image of the confusion matrix.
    """
    # Load the confusion matrix
    confusion_matrix = np.load(npy_file_path)
    
    # Convert the confusion matrix to percentages
    confusion_matrix_percentage = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot using seaborn for a nicer display
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix_percentage, annot=True, fmt=".2%", cmap='Blues', cbar_kws={'format': '%.0f%%'})
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Percentage)')
    
    # Save the plot
    plt.savefig(output_file_path)
    
    # Show the plot
    # plt.show()

# Example usage:
plot_and_save_confusion_matrix_percentage('/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results_metric_output/confusion_matrix_multiclassmorevalues_true.npy', '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/confusion_matrix_true')
