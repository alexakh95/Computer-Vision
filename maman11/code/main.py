import detector_desc_helper as hp
import detector_desc as detector_desc
from pathlib import Path
# Navigate to the main folder and locate the images directory
main_folder = Path(__file__).resolve().parents[1]  # Get the parent folder of 'code'
path = main_folder / 'images'
detector = detector_desc.DetectorDescriptor(str(path))
#load images
detector.load_images()

#Lets calculate the repeatability and positioning error for each detector and transformation
results_harris_rot = hp.rotation_test(detector,detector_desc.Matcher(),detector.harris_detector, 30)
results_harris_rot_70 = hp.rotation_test( detector,detector_desc.Matcher(),detector.harris_detector, 70)
results_harris_scale_05 = hp.scaled_test(detector, detector_desc.Matcher(),detector.harris_detector, 0.5)
results_harris_scale_2 = hp.scaled_test(detector, detector_desc.Matcher(),detector.harris_detector, 2.0)
results_harris_blur = hp.blur_test(detector,detector_desc.Matcher(),detector.harris_detector)    
results_harris_gaussian = hp.gaussian_noise_test(detector, detector_desc.Matcher(),detector.harris_detector)

results_orb_rot = hp.rotation_test(detector, detector_desc.Matcher(),detector.orb_detector, 30, use_descriptors=True)
results_orb_rot_70 = hp.rotation_test(detector,detector_desc.Matcher(),detector.orb_detector, 70,use_descriptors=True)
results_orb_scale_05 = hp.scaled_test(detector, detector_desc.Matcher(),detector.orb_detector, 0.5,use_descriptors=True)
results_orb_scale_2 = hp.scaled_test(detector, detector_desc.Matcher(),detector.orb_detector, 2.0,use_descriptors=True)
results_orb_blur = hp.blur_test(detector,detector_desc.Matcher(),detector.orb_detector,use_descriptors=True)
results_orb_gaussian = hp.gaussian_noise_test(detector,detector_desc.Matcher(),detector.orb_detector,use_descriptors=True)

results_sift_rot =hp.rotation_test(detector,detector_desc.Matcher(),detector.sift_detector,30,use_descriptors=True)
results_sift_rot_70 = hp.rotation_test(detector, detector_desc.Matcher(),detector.sift_detector, 70,use_descriptors=True)
results_sift_scale_05 = hp.scaled_test( detector,detector_desc.Matcher(),detector.sift_detector, 0.5,use_descriptors=True) 
results_sift_scale_2 = hp.scaled_test(detector, detector_desc.Matcher(),detector.sift_detector, 2.0,use_descriptors=True)
results_sift_blur = hp.blur_test(detector,detector_desc.Matcher(),detector.sift_detector,use_descriptors=True)
results_sift_gaussian = hp.gaussian_noise_test(detector, detector_desc.Matcher(),detector.sift_detector,use_descriptors=True)

results_fast_rot = hp.rotation_test(detector,detector_desc.Matcher(),detector.fast_detector, 30)
results_fast_rot_70 = hp.rotation_test(detector, detector_desc.Matcher(),detector.fast_detector, 70)
results_fast_scale_05 = hp.scaled_test(detector, detector_desc.Matcher(),detector.fast_detector, 0.5)
results_fast_scale_2 = hp.scaled_test(detector,detector_desc.Matcher(),detector.fast_detector, 2.0)
results_fast_blur = hp.blur_test(detector,detector_desc.Matcher(),detector.fast_detector)
results_fast_gaussian = hp.gaussian_noise_test(detector,detector_desc.Matcher(),detector.fast_detector)


results_akaze_rot = hp.rotation_test(detector,detector_desc.Matcher(),detector.akaze_descriptor, 30,use_descriptors=True)
results_akaze_rot_70 = hp.rotation_test(detector,detector_desc.Matcher(),detector.akaze_descriptor, 70,use_descriptors=True)
results_akaze_scale_05 = hp.scaled_test(detector,detector_desc.Matcher(),detector.akaze_descriptor, 0.5,use_descriptors=True)
results_akaze_scale_2 = hp.scaled_test(detector,detector_desc.Matcher(),detector.akaze_descriptor, 2.0,use_descriptors=True)
results_akaze_blur = hp.blur_test(detector,detector_desc.Matcher(),detector.akaze_descriptor,use_descriptors=True)
results_akaze_gaussian = hp.gaussian_noise_test(detector,detector_desc.Matcher(),detector.akaze_descriptor,use_descriptors=True)



#TODO: plot the results
import matplotlib.pyplot as plt
import numpy as np
transformations = ['Rotation 30', 'Rotation 70', 'Scale 0.5', 'Scale 2.0', 'Blur', 'Gaussian Noise']

detectors = ['Harris', 'ORB', 'SIFT', 'FAST']
descriptors = ['ORB', 'SIFT','AKAZE']

rep_harris = [results_harris_rot['repeatability'],results_harris_rot_70['repeatability'],results_harris_scale_05['repeatability'],results_harris_scale_2['repeatability'],results_harris_blur['repeatability'],results_harris_gaussian['repeatability']]
rep_orb = [results_orb_rot['repeatability'],results_orb_rot_70['repeatability'],results_orb_scale_05['repeatability'],results_orb_scale_2['repeatability'],results_orb_blur['repeatability'],results_orb_gaussian['repeatability']]
rep_sift = [results_sift_rot['repeatability'],results_sift_rot_70['repeatability'],results_sift_scale_05['repeatability'],results_sift_scale_2['repeatability'],results_sift_blur['repeatability'],results_sift_gaussian['repeatability']]
rep_fast = [results_fast_rot['repeatability'],results_fast_rot_70['repeatability'],results_fast_scale_05['repeatability'],results_fast_scale_2['repeatability'],results_fast_blur['repeatability'],results_fast_gaussian['repeatability']]

rep = [rep_harris,rep_orb,rep_sift,rep_fast]

# Set bar width and x positions
bar_width = 0.2
x = np.arange(len(transformations))  # Positions for the transformations

#Plot each detector
plt.figure(figsize=(10, 6))
for i, detector in enumerate(detectors):
    plt.bar(x + i * bar_width, rep[i], width=bar_width, label=detector)
    
# Add labels and legend
plt.xlabel("Transformation", fontsize=12)
plt.ylabel("Repeatability", fontsize=12)
plt.title("Average Repeatability by Detector and Transformation", fontsize=14)
plt.xticks(x + bar_width * (len(detectors) - 1) / 2, transformations, rotation=45)
plt.legend(title="Detector", fontsize=10)
plt.tight_layout()

# Show the plot
plt.show(block = True)

#plot positioning error
error_harris = [results_harris_rot['positioning_error'],results_harris_rot_70['positioning_error'],results_harris_scale_05['positioning_error'],results_harris_scale_2['positioning_error'],results_harris_blur['positioning_error'],results_harris_gaussian['positioning_error']]
error_orb = [results_orb_rot['positioning_error'],results_orb_rot_70['positioning_error'],results_orb_scale_05['positioning_error'],results_orb_scale_2['positioning_error'],results_orb_blur['positioning_error'],results_orb_gaussian['positioning_error']]
error_sift = [results_sift_rot['positioning_error'],results_sift_rot_70['positioning_error'],results_sift_scale_05['positioning_error'],results_sift_scale_2['positioning_error'],results_sift_blur['positioning_error'],results_sift_gaussian['positioning_error']]
error_fast = [results_fast_rot['positioning_error'],results_fast_rot_70['positioning_error'],results_fast_scale_05['positioning_error'],results_fast_scale_2['positioning_error'],results_fast_blur['positioning_error'],results_fast_gaussian['positioning_error']]

errors = [error_harris,error_orb,error_sift,error_fast]
#Plot each detector
plt.figure(figsize=(10, 6))
for i, detector in enumerate(detectors):
    plt.bar(x + i * bar_width, errors[i], width=bar_width, label=detector)
    
# Add labels and legend
plt.xlabel("Transformation", fontsize=12)
plt.ylabel("Positioning Error", fontsize=12)
plt.title("Average Positioning Error by Detector and Transformation", fontsize=14)
plt.xticks(x + bar_width * (len(detectors) - 1) / 2, transformations, rotation=45)
plt.legend(title="Detector", fontsize=10)
plt.tight_layout()

# Show the plot
plt.show(block=True)

#plot descriptors accuracy 
descriptors = ['ORB', 'SIFT','AKAZE']
accuracy_sift = [results_sift_rot['matching_accuracy'],results_sift_rot_70['matching_accuracy'],results_sift_scale_05['matching_accuracy'],results_sift_scale_2['matching_accuracy'],results_sift_blur['matching_accuracy'],results_sift_gaussian['matching_accuracy']]
accuracy_akaze = [results_akaze_rot['matching_accuracy'],results_akaze_rot_70['matching_accuracy'],results_akaze_scale_05['matching_accuracy'],results_akaze_scale_2['matching_accuracy'],results_akaze_blur['matching_accuracy'],results_akaze_gaussian['matching_accuracy']]
accuracy_orb = [results_orb_rot['matching_accuracy'],results_orb_rot_70['matching_accuracy'],results_orb_scale_05['matching_accuracy'],results_orb_scale_2['matching_accuracy'],results_orb_blur['matching_accuracy'],results_orb_gaussian['matching_accuracy']]

accuracy = [accuracy_orb,accuracy_sift,accuracy_akaze]

#Plot each detector
plt.figure(figsize=(10, 6))
for i, detector in enumerate(descriptors):
    plt.bar(x + i * bar_width, accuracy[i], width=bar_width, label=detector)
    
# Add labels and legend
plt.xlabel("Transformation", fontsize=12)
plt.ylabel("Matching Accuracy", fontsize=12)
plt.title("Average Matching Accuracy by Descriptor and Transformation", fontsize=14)
plt.xticks(x + bar_width * (len(detectors) - 1) / 2, transformations, rotation=45)
plt.legend(title="Descriptor", fontsize=10)
plt.tight_layout()
# Show the plot
plt.show(block=True)

time_trans_sift = [np.mean(results_sift_rot['time_trans']),np.mean(results_sift_rot_70['time_trans']),np.mean(results_sift_scale_05['time_trans']),np.mean(results_sift_scale_2['time_trans']),np.mean(results_sift_blur['time_trans']),np.mean(results_sift_gaussian['time_trans'])]
time_trans_akaze = [np.mean(results_akaze_rot['time_trans']),np.mean(results_akaze_rot_70['time_trans']),np.mean(results_akaze_scale_05['time_trans']),np.mean(results_akaze_scale_2['time_trans']),np.mean(results_akaze_blur['time_trans']),np.mean(results_akaze_gaussian['time_trans'])]
time_trans_orb = [np.mean(results_orb_rot['time_trans']),np.mean(results_orb_rot_70['time_trans']),np.mean(results_orb_scale_05['time_trans']),np.mean(results_orb_scale_2['time_trans']),np.mean(results_orb_blur['time_trans']),np.mean(results_orb_gaussian['time_trans'])]
time_trans = [time_trans_orb,time_trans_sift,time_trans_akaze]

#Plot each detector
plt.figure(figsize=(10, 6))
for i, detector in enumerate(descriptors):
    plt.bar(x + i * bar_width, time_trans[i], width=bar_width, label=detector)
    
# Add labels and legend
plt.xlabel("Transformation", fontsize=12)
plt.ylabel("Calculation Time", fontsize=12)
plt.title("Average Calculation Time by Descriptor and Transformation", fontsize=14)
plt.xticks(x + bar_width * (len(detectors) - 1) / 2, transformations, rotation=45)
plt.legend(title="Descriptor", fontsize=10)
plt.tight_layout()
# Show the plot
plt.show(block=True)



