import pipelines_utils as pput
import utility as ut

if __name__ == "__main__":
    dataset = ut.ImageDataset()
    dataset.parse_labels_from_file_name()

    print("Running VGG pipeline...")
    pput.vgg_pipeline(dataset)

    print("Running SIFT pipeline...")
    pput.sift_pipeline(dataset)
