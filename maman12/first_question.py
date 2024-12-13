import numpy as np 
import pandas as pd 
import os
import cv2
from sklearn.model_selection import train_test_split 
from pathlib import Path

class ImageDataset:
    def __init__(self):
        self.labels = None
        self.image_paths = None
    
    def parse_labels_from_file_name(self,folder_name: str = 'image_dataset'):
    
        """
        Parses image file names in the specified folder to extract labels and image paths.
        This method assumes that the label is the part of the file name before the first underscore.
        It supports image files with extensions '.jpg', '.png', and '.jpeg'.
        Args:
            folder_name (str): The name of the folder containing the images. Defaults to 'image_dataset'.
        Attributes:
            labels (list): A list of labels extracted from the image file names.
            image_paths (list): A list of full paths to the image files.
    
        """
        
        main_folder = Path(__file__).resolve().parents[0]  # Get the parent folder of 'code'
        path = main_folder / folder_name       
        
        # List all image paths and extract labels
        image_paths = []
        labels = []

        for image_name in os.listdir(path):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Adjust for your file types
                label = image_name.split('_')[0]  # Assumes label is before the first underscore
                image_paths.append(os.path.join(path, image_name))
                labels.append(label)

        self.labels = labels
        self.image_paths = image_paths
        print(f"Loaded {len(image_paths)} images with labels.")

    def load_images(self, list_dir):
        """
        Load images from a list of directories.
        Args:
            list_dir (list): A list of directories containing images.
        Returns:
            images (np.ndarray): A numpy array of images.
        """
        images = []
        for dir in list_dir:
            image = cv2.imread(dir)
            images.append(image)
            
        return np.array(images)

    