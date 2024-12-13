from sklearn.metrics import accuracy_score
import first_question as fq
from sklearn.model_selection import train_test_split
import cv2
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    kmeans = KMeans(n_clusters=100,n_init=2, random_state=42)
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
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.2f}")
    return clf


dataset = fq.ImageDataset()
dataset.parse_labels_from_file_name()


# Split into train+val and test
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    dataset.image_paths, dataset.labels, test_size=0.2, stratify=dataset.labels, random_state=42
)

# Split train+val into train and validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.25, stratify=train_val_labels, random_state=42
)

print(f"Train set: {len(train_paths)} images")
print(f"Validation set: {len(val_paths)} images")
print(f"Test set: {len(test_paths)} images")

#load the images from the directory
train_images = dataset.load_images(train_paths)

#get the keypoints and descriptors for each image in train set
_ , train_des = sift_detect_describe(train_images)
all_train_descriptors=build_descriptor_array(train_des)
vocabulary = vec_quantization(all_train_descriptors)

#get the descriptors for each image in the validation set
val_images = dataset.load_images(val_paths)
_ , val_des = sift_detect_describe(val_images)


#create histograms for each image and label in the training set
train_histograms = build_histograms(train_des, vocabulary)
valid_histograms = build_histograms(val_des, vocabulary)

#train the random forest classifier
random_forest_classifier(train_histograms, train_labels, valid_histograms, val_labels)

print("Vocabulary created")





