import utility as ut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


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
train_images = dataset.load_images(train_paths)

#get the keypoints and descriptors for each image in train set
_ , train_des = ut.sift_detect_describe(train_images)
all_train_descriptors=ut.build_descriptor_array(train_des)
vocabulary = ut.vec_quantization(all_train_descriptors)

#get the descriptors for each image in the validation set
val_images = dataset.load_images(val_paths)
_ , val_des = ut.sift_detect_describe(val_images)

#test 
test_images = dataset.load_images(test_paths)
_ , test_des = ut.sift_detect_describe(test_images)

#create histograms for each image and label in the training set
train_histograms = ut.build_histograms(train_des, vocabulary)
valid_histograms = ut.build_histograms(val_des, vocabulary)
test_histograms = ut.build_histograms(test_des, vocabulary)

#train the random forest classifier
_,test_predict, y_score = ut.random_forest_classifier(train_histograms, train_labels, test_histograms, test_labels)


#build precision and recall curve for each class
y_test_binarized = label_binarize(test_labels, classes=dataset.unique_labels)

ut.plot_precision_recall_curve(y_test_binarized, y_score, dataset.unique_labels)
ut.plot_roc_curve(y_test_binarized, y_score, dataset.unique_labels)
ut.plot_confusion_matrix(test_labels, test_predict, dataset.unique_labels)










