import numpy as np
def calc_rep_and_error(mod_kp, origin_kp,matcher,threshold):
    
    """
    Calculate repeatability and positioning error between modified and original keypoints.
    Parameters:
        mod_kp (list): List of modified keypoints.
        origin_kp (list): List of original keypoints.
        matcher (object): An object that provides methods to match keypoints and calculate metrics.
        threshold (float): Threshold value for matching keypoints.
    Returns:
        tuple: A tuple containing the mean repeatability and mean positioning error.
    """
    
    #get the matches between the keypoints, calculate repeatability and positioning error
    calculate_rep = []
    calc_pos_error = []
    for i in range(len(mod_kp)):
        matches = matcher.match_keypoints(origin_kp[i], mod_kp[i], threshold)
        calculate_rep.append(matcher.calculate_repeatability(origin_kp[i], mod_kp[i], matches, threshold))
        calc_pos_error.append(matcher.calculate_positioning_error(origin_kp[i], mod_kp[i], matches))

    return np.mean(calculate_rep), np.mean(calc_pos_error)


def calc_matching_accuracy(origin_kp,mod_kp,origin_des,mod_des,matcher,threshold):
    
    """
    Calculate the matching accuracy between original and modified keypoints and descriptors.
        Args:
            origin_kp (list): List of keypoints from the original image.
            mod_kp (list): List of keypoints from the modified image.
            origin_des (list): List of descriptors from the original image.
            mod_des (list): List of descriptors from the modified image.
            matcher (object): An object that provides methods to match descriptors and calculate matching accuracy.
            threshold (float): Threshold value to determine a match.
    Returns:
            float: The mean matching accuracy between the original and modified descriptors.
    """
    #get the matches between the keypoints, calculate matching error and speed
    matching_accuracy = []
    for i in range(len(mod_kp)):
        match = matcher.match_descriptors(origin_des[i], mod_des[i])
        if len(match) < 4:
            # Sort matches by distance
            matches = sorted(match, key=lambda x: x.distance)
            # Apply a distance threshold (e.g., 50)
            good_matches = [m for m in matches if m.distance < threshold]
            matching_accuracy.append(len(good_matches) / len(matches))
        else:
            matching_accuracy.append(matcher.calculate_matching_accuracy(origin_kp[i], mod_kp[i], match ,threshold))
       
    return np.mean(matching_accuracy)

    
def test_template(matcher,detec_func, transform_function, threshold_function,use_descriptors, keypoint_adjustment=None, transform_params=None):
    
    """
        Test template function for evaluating keypoint detection and descriptor matching.
        Parameters:
        matcher (object): Matcher object used for matching keypoints or descriptors.
        detec_func (callable): Function to detect keypoints or descriptors.
        transform_function (callable): Function to apply transformations to images.
        threshold_function (callable): Function to calculate the threshold for matching.
        use_descriptors (bool): Flag to indicate whether to use descriptors.
        keypoint_adjustment (callable, optional): Function to adjust keypoints (e.g., scaling or rotation). Defaults to None.
        transform_params (dict, optional): Parameters for the transformation function. Defaults to None.
        Returns:
        dict: A dictionary containing the following keys:
            - "repeatability" (float): The repeatability score of keypoints.
            - "positioning_error" (float): The positioning error of keypoints.
            - "time_trans" (float): The time taken for transformation.
            - "matching_accuracy" (float or None): The matching accuracy of descriptors, if applicable.
        """
    
    # Apply transformation to the images
    if transform_params:
        transformed_images = transform_function(transform_params)
    else:
        transformed_images = transform_function()
    
    # Detect keypoints or descriptors in original and transformed images
    if use_descriptors:
        origin_kp, origin_desc, _ = detec_func()
        transformed_kp, transformed_desc,time_trans = detec_func(transformed_images)   
    else:
        origin_kp = detec_func()
        transformed_kp = detec_func(transformed_images)
        origin_desc = transformed_desc =time_trans= None
        
        
    # Calculate the threshold
    threshold = threshold_function(transformed_images[0].shape)
    
         
    # Adjust original keypoints if required (e.g., scaling or rotation)
    if keypoint_adjustment:
        origin_kp = [keypoint_adjustment(kp) for kp in origin_kp]
    
    calculate_rep, calc_pos_error = calc_rep_and_error(transformed_kp, origin_kp, matcher, threshold) 
      
    matching_accuracy = None
      
    if use_descriptors and origin_desc is not None and transformed_desc is not None :
        matching_accuracy = calc_matching_accuracy(origin_kp, transformed_kp,origin_desc,transformed_desc,matcher,threshold)
        
    return  {
        "repeatability": calculate_rep,
        "positioning_error": calc_pos_error,
        "time_trans": time_trans,
        "matching_accuracy": matching_accuracy,
    }


def rotation_test(detector,matcher,detec_func, angle,use_descriptors=False):
    
    """
    Tests the performance of a detector under image rotation.

    Parameters:
        detector (object): An object that contains methods for image rotation and keypoint adjustment.
        matcher (object): An object that contains methods for matching keypoints and calculating thresholds.
        detec_func (callable): A function to detect features in the image.
        angle (float): The angle by which to rotate the images.
        use_descriptors (bool, optional): Whether to use descriptors in the test. Defaults to False.

    Returns:
        result: The dictionary results of the test_template function, which evaluates the detector's performance.
    """
    return test_template(
        matcher,
        detec_func=detec_func,
        transform_function=detector.rotate_images,
        threshold_function=lambda shape :matcher.calculate_threshold(shape),
        keypoint_adjustment=lambda kp: matcher.rotate_keypoints(kp, detector.rotation_matrix),
        transform_params=angle,
        use_descriptors=use_descriptors
    )

def scaled_test( detector,matcher,detec_func,scale,use_descriptors=False):
    """
    Tests the performance of a detector under image scaling.

    Parameters:
        detector (object): An object that contains methods for image rotation and keypoint adjustment.
        matcher (object): An object that contains methods for matching keypoints and calculating thresholds.
        detec_func (callable): A function to detect features in the image.
        scale (float): The scale by which to scale the images.
        use_descriptors (bool, optional): Whether to use descriptors in the test. Defaults to False.

    Returns:
        result: The dictionary results of the test_template function, which evaluates the detector's performance.
    """
    return test_template(
        matcher,
        detec_func=detec_func,
        transform_function=detector.scale_images,
        threshold_function=lambda img: matcher.calculate_scaled_threshold(img, 0),
        keypoint_adjustment=lambda kp: matcher.adjust_keypoints_for_scale(kp, scale),
        transform_params=scale,
        use_descriptors=use_descriptors,
    )
def blur_test(detector,matcher,detec_func,use_descriptors=False):
    """
        Tests the performance of a detector and matcher under Gaussian blur transformation.
        Args:
            detector: An object that has a `gaussian_blur` method for applying Gaussian blur.
            matcher: An object that has a `calculate_threshold` method for calculating the threshold.
            detec_func: A function to be used for detection.
            use_descriptors (bool, optional): A flag indicating whether to use descriptors. Defaults to False.
        Returns:
            The result of the `test_template` function with the provided parameters.
    """
    return test_template(
        matcher,
        detec_func=detec_func,
        transform_function=detector.gaussian_blur,
        threshold_function=lambda shape :matcher.calculate_threshold(shape),
        use_descriptors=use_descriptors
        
    )

def gaussian_noise_test(detector, matcher, detec_func, use_descriptors=False):
    """
    Tests the performance of a detector and descriptor under Gaussian noise transformation.

    Args:
        detector: An object that has a `gaussian_noise` method for applying Gaussian noise.
        matcher: An object that has a `calculate_threshold` method for calculating the threshold.
        detec_func: A function to be used for detection.
        use_descriptors (bool, optional): A flag indicating whether to use descriptors. Defaults to False.

    Returns:
        The result of the `test_template` function with the provided parameters.
    """
    return test_template(
        matcher,
        detec_func=detec_func,
        transform_function=detector.gaussian_noise,
        threshold_function=lambda shape :matcher.calculate_threshold(shape),
        use_descriptors=use_descriptors
    )