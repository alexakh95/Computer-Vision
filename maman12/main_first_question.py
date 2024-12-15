from sklearn.metrics import  precision_recall_curve, average_precision_score
import utilities as fq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

dataset = fq.ImageDataset()
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
_ , train_des = fq.sift_detect_describe(train_images)
all_train_descriptors=fq.build_descriptor_array(train_des)
vocabulary = fq.vec_quantization(all_train_descriptors)

#get the descriptors for each image in the validation set
val_images = dataset.load_images(val_paths)
_ , val_des = fq.sift_detect_describe(val_images)

#test 
test_images = dataset.load_images(test_paths)
_ , test_des = fq.sift_detect_describe(test_images)

#create histograms for each image and label in the training set
train_histograms = fq.build_histograms(train_des, vocabulary)
valid_histograms = fq.build_histograms(val_des, vocabulary)
test_histograms = fq.build_histograms(test_des, vocabulary)

#train the random forest classifier
_,test_predict, y_score = fq.random_forest_classifier(train_histograms, train_labels, test_histograms, test_labels)


#build precision and recall curve for each class
n_classes = len(dataset.unique_labels)
y_test_binarized = label_binarize(test_labels, classes=dataset.unique_labels)

plt.figure(figsize=(10, 7))

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
    avg_precision = average_precision_score(y_test_binarized[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f"{dataset.unique_labels[i]} (AP={avg_precision:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Class")
plt.legend(loc="best")
plt.grid()
plt.show(block=True)


#build ROC curve for each class
from sklearn.metrics import roc_curve, auc

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{dataset.unique_labels[i]} (AUC={roc_auc:.2f})")
    
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Each Class")
plt.legend(loc="best")
plt.grid()
plt.show(block=True)

#build confusion matrix for the test set
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.figure(figsize=(10, 10))
conf_matrix = confusion_matrix(test_labels, test_predict, labels=dataset.unique_labels)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=dataset.unique_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show(block=True)





