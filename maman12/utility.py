import numpy as np 
import os
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_curve,average_precision_score,roc_curve,auc
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import torch
from  torchvision import transforms 
import torchvision.models as  models 
from PIL import Image
import matplotlib.pyplot as plt

class ImageDataset:
    def __init__(self):
        self.labels = None
        self.unique_labels = None
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
        self.unique_labels = np.unique(self.labels)
        print(f"Loaded {len(image_paths)} images with labels.")
        print(f"Found {len(self.unique_labels)} unique labels.{self.unique_labels}")
        
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
    
    def load_image_as_tensor(self,image_path):
        images = []
        for path in image_path:
            # Define a transform to convert the image to a tensor
            transform = transforms.Compose([
                transforms.ToTensor(),# Convert the image to a tensor
                transforms.Normalize( # Normalize to ImageNet mean and std
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Open the image file
            image = Image.open(path).convert("RGB")
            # Apply the transform to the image
            image_tensor = transform(image)
            
            images.append(image_tensor)
    
        return torch.stack(images)

def sift_detect_describe(images):
    """
    Detect and describe keypoints in images using SIFT.
    Args:
        images (np.ndarray): A numpy array of images.
    Returns:
        keypoints (list): A list of keypoints.
        descriptors (list): A list of descriptors.
    """
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(np.array(des,dtype=np.float32))
    return keypoints, descriptors

def build_descriptor_array(descriptors):
    """
    Concatenate all descriptors into a single 2D numpy array.
    Args:
        descriptors (list of np.ndarray): A list of descriptor arrays for each image.
    Returns:
        all_descriptors (np.ndarray): A 2D numpy array of shape (total_features, feature_length).
    """
    # Filter out None and empty arrays
    valid_descriptors = [des for des in descriptors if des is not None and len(des) > 0]
    
    # Concatenate all descriptors into a 2D array
    all_descriptors = np.vstack(valid_descriptors)
    
    return all_descriptors


def vec_quantization(descriptors):
    """
    Quantize descriptors using K-means clustering.
    Args:
        descriptors (list): A list of descriptors.
    Returns:
        codebook (np.ndarray): A numpy array of centroids.
    """
    kmeans = KMeans(n_clusters=100,n_init=1, random_state=42)
    kmeans.fit(descriptors)
    
    return kmeans.cluster_centers_

def build_histograms(descriptors, codebook):
    """
    Build histograms of visual words.
    Args:
        descriptors (list): A list of descriptors.
        codebook (np.ndarray): A numpy array of centroids.
    Returns:
        histograms (list): A list of histograms.
    """
    histograms = []
    for des in descriptors:
        histogram = np.zeros(len(codebook))
        for d in des:
            idx = np.argmin(np.linalg.norm(codebook - d, axis=1))
            histogram[idx] += 1
        histograms.append(histogram)
    return histograms

def random_forest_classifier(X_train, y_train, X_val, y_val):
    """
    Train a random forest classifier and evaluate on the validation set.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
    Returns:
        clf (RandomForestClassifier): Trained random forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.2f}")
    return clf, y_pred, y_proba



def cnn_features(images):
    """
    Extracts feature maps from the last convolutional layer of a pre-trained VGG-16 model.
    Args:
        images (torch.Tensor): A batch of images to extract features from. The tensor should be of shape 
                               (batch_size, channels, height, width) and normalized appropriately.
    Returns:
        torch.Tensor: The feature maps extracted from the last convolutional layer of the VGG-16 model.
                      The shape of the returned tensor will depend on the input image dimensions.
    """
    
    # Load a pre-trained vgg-16 model
    model = models.vgg16(pretrained=True)
    # Remove the classifier (fully connected layers) to get the feature map from the last conv layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    # Set the feature extractor to evaluation mode
    feature_extractor.eval()
    
    with torch.no_grad():  # Disable gradient calculation for inference
        feat_map = feature_extractor(images)
    
    return feat_map
    
    
def plot_confusion_matrix(test_labels, test_predict, unique_labels):

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    plt.figure(figsize=(10, 10))
    conf_matrix = confusion_matrix(test_labels, test_predict, labels=unique_labels)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=unique_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show(block=True)
    
    
def plot_precision_recall_curve(y_test_binarized, y_score,unique_labels):
    plt.figure(figsize=(10, 7))

    for i in range(len(unique_labels)):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
        avg_precision = average_precision_score(y_test_binarized[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"{unique_labels[i]} (AP={avg_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Each Class")
    plt.legend(loc="best")
    plt.grid()
    plt.show(block=True)
    
def plot_roc_curve( y_test_binarized, y_score, unique_labels):
    for i in range(len(unique_labels)):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{unique_labels[i]} (AUC={roc_auc:.2f})")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Each Class")
    plt.legend(loc="best")
    plt.grid()
    plt.show(block=True)