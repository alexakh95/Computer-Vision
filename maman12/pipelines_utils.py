import utility as ut
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def split_dataset(dataset, test_size=0.4, val_test_ratio=0.5):
    """
    Split the dataset into train, validation, and test sets.
    """
    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        dataset.image_paths, dataset.labels, test_size=test_size, stratify=dataset.labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remaining_paths, remaining_labels, test_size=val_test_ratio, stratify=remaining_labels, random_state=42
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def extract_vgg_features(paths, dataset):
    """
    Extract CNN features using a pretrained VGG-16 model.
    """
    images = dataset.load_image_as_tensor(paths)
    features = ut.cnn_features(images)
    return torch.flatten(features, start_dim=1)


def extract_sift_features(paths, dataset):
    """
    Extract SIFT keypoints and descriptors for a set of images.
    """
    images = dataset.load_images(paths)
    _, descriptors = ut.sift_detect_describe(images)
    return descriptors


def build_histograms_for_sets(train_descriptors, val_descriptors, test_descriptors, vocabulary, flatten_features=False):
    """
    Build histograms for train, validation, and test datasets.
    """
    train_histograms = ut.build_histograms(train_descriptors, vocabulary, flatten_features)
    val_histograms = ut.build_histograms(val_descriptors, vocabulary, flatten_features)
    test_histograms = ut.build_histograms(test_descriptors, vocabulary, flatten_features)
    return train_histograms, val_histograms, test_histograms


def train_and_evaluate(train_histograms, train_labels, test_histograms, test_labels, unique_labels):
    """
    Train a Random Forest classifier and evaluate using precision-recall and ROC curves.
    """
    _, test_predict, y_score = ut.random_forest_classifier(train_histograms, train_labels, test_histograms, test_labels)
    
    y_test_binarized = label_binarize(test_labels, classes=unique_labels)
    
    ut.plot_precision_recall_curve(y_test_binarized, y_score, unique_labels)
    ut.plot_roc_curve(y_test_binarized, y_score, unique_labels)
    ut.plot_confusion_matrix(test_labels, test_predict, unique_labels)


def vgg_pipeline(dataset):
    """
    VGG-16 based feature extraction pipeline.
    """
    # Dataset splitting
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(dataset)

    # Feature extraction
    train_features = extract_vgg_features(train_paths, dataset)
    val_features = extract_vgg_features(val_paths, dataset)
    test_features = extract_vgg_features(test_paths, dataset)

    # Vocabulary building
    vocabulary = ut.vec_quantization(train_features.detach().numpy())

    # Histogram building
    train_histograms, val_histograms, test_histograms = build_histograms_for_sets(
        train_features, val_features, test_features, vocabulary, flatten_features=True
    )

    # Train and evaluate
    train_and_evaluate(train_histograms, train_labels, test_histograms, test_labels, dataset.unique_labels)


def sift_pipeline(dataset):
    """
    SIFT-based feature extraction pipeline.
    """
    # Dataset splitting
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(dataset)

    # Feature extraction
    train_descriptors = extract_sift_features(train_paths, dataset)
    all_train_descriptors = ut.build_descriptor_array(train_descriptors)
    vocabulary = ut.vec_quantization(all_train_descriptors)

    val_descriptors = extract_sift_features(val_paths, dataset)
    test_descriptors = extract_sift_features(test_paths, dataset)

    # Histogram building
    train_histograms, val_histograms, test_histograms = build_histograms_for_sets(
        train_descriptors, val_descriptors, test_descriptors, vocabulary
    )

    # Train and evaluate
    train_and_evaluate(train_histograms, train_labels, test_histograms, test_labels, dataset.unique_labels)
