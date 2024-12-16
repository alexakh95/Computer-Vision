
import utility as ut
import torch
from sklearn.model_selection import train_test_split
dataset = ut.ImageDataset()
dataset.parse_labels_from_file_name()

# Split into train (60%) and remaining (40%)
train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
    dataset.image_paths, dataset.labels, test_size=0.4, stratify=dataset.labels, random_state=42
)

# Split remaining into validation (50% of remaining, which is 20% of total) and test (50% of remaining, which is 20% of total)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    remaining_paths, remaining_labels, test_size=0.5, stratify=remaining_labels, random_state=42
)

#load the images from the directory
train_images =dataset.load_image_as_tensor(train_paths)
val_images = dataset.load_image_as_tensor(val_paths)
test_images = dataset.load_image_as_tensor(test_paths)

#extract features from the images using a pretrained model
train_features = ut.cnn_features(train_images)
val_features = ut.cnn_features(val_images)
test_features = ut.cnn_features(test_images)

#to use kmeans we need to flatten our tensor features to 2D before clustering
#the shape are (batch_size,512,8,8) so after flattening we get (batch_size,512*8*8)
train_features = torch.flatten(train_features,start_dim = 1)
val_features = torch.flatten(val_features,start_dim = 1)
test_features = torch.flatten(test_features,start_dim = 1)


#get vocabulary using kmeans clustering 
vocabulary = ut.vec_quantization(train_features.detach().numpy())

#build histograms for each image and label in the training set
train_histograms = ut.build_histograms(train_features, vocabulary)
valid_histograms = ut.build_histograms(val_features, vocabulary)
test_histograms = ut.build_histograms(test_features, vocabulary)

#train the random forest classifier
_,test_predict, y_score = ut.random_forest_classifier(train_histograms, train_labels, test_histograms, test_labels)

ut.plot_precision_recall_curve(test_labels, y_score, dataset.unique_labels)
ut.plot_roc_curve(test_labels, y_score, dataset.unique_labels)
ut.plot_confusion_matrix(test_labels, test_predict, dataset.unique_labels)




