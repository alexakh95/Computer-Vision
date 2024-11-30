import cv2
import numpy as np
import os
import time


class Image:
    """ 
    The Image class provides methods to load, manipulate, and process images from a specified directory path.
    Attributes:
        __path (str): The directory path where images are stored.
        images (list): A list to store loaded images.
        rotation_matrix (numpy.ndarray): The rotation matrix used for rotating images.
    Methods:
        __init__(path):
            Initializes the Image object with the specified directory path.
        load_images():
        rotate_images(angle):
        scale_images(scale):
        gaussian_blur():
        gaussian_noise():
        get_path():
            Returns the directory path where images are stored.
        get_images():
            Returns the list of loaded images.
    """
    def __init__(self,path):
        self.__path = path
        self.images = []
        self.rotation_matrix = None
    
    def load_images(self):
        """
        Loads images from the specified directory path and appends them to the images list.

        This method iterates through all files in the directory specified by self.path,
        reads each image using OpenCV, and appends the loaded image to the self.images list.

        Raises:
            FileNotFoundError: If the specified directory path does not exist.
            cv2.error: If an image file cannot be read by OpenCV.
        """
        for img in os.listdir(self.__path):
            img_path = os.path.join(self.__path, img)
            image = cv2.imread(img_path)
            if image is not None:
                self.images.append(image)
            
    
    def rotate_images(self,angle):
        """
        Rotates each image in the self.images list by the specified angle.
        Parameters:
        angle (float): The angle by which to rotate the images, in degrees.
        Returns:
        list: A list of rotated images.
        """
        
        images = np.copy(self.images)
        rotated_images = []  
        for image in images:
            # Get the height and width of the image
            (h, w) = image.shape[:2]

            # Compute the center of the image
            center = (w // 2, h // 2)

            # Get the rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            self.rotation_matrix = M
            
            # Apply the rotation without changing the shape
            rotated = cv2.warpAffine(image, M, (w, h))
            rotated_images.append(rotated)
        
        return rotated_images
    
    def scale_images(self,scale):
        """
        Scales the images by a given factor.

        Parameters:
        scale (float): The scaling factor to resize the images.

        Returns:
        list: A list of scaled images.
        """
        scaled_images = []
        images = np.copy(self.images)
        for img in images:
            # Resize the image
            scaled_image = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled_images.append(scaled_image)      
        return scaled_images
    
    def gaussian_blur(self):
        """
        Applies Gaussian blur to each image in the self.images list.
        This method creates a blurred version of each image using a Gaussian filter
        with a kernel size of 5x5 and a standard deviation of 0. The blurred images
        are stored in a new list and returned.
        Returns:
            list: A list of blurred images.
        """
        blur_images = []
        images = np.copy(self.images)
        for img in images:
            
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            blur_images.append(blurred)
        
        return blur_images
    
    def gaussian_noise(self):   
        """
        Applies Gaussian noise to each image in the self.images list.
        This method generates Gaussian noise with a mean of 0 and a standard deviation of 50,
        and adds this noise to each image in the self.images list. The resulting noisy images
        are returned as a new list.
        Returns:
            list: A list of images with Gaussian noise added.
        """
        noisy_images = []
        images = np.copy(self.images)
        for img in images:
            # Generate Gaussian noise
            noise = np.random.normal(loc=0, scale=50, size=img.shape).astype(np.uint8)
            noisy = cv2.add(img, noise)
            noisy_images.append(noisy)
        
        return noisy_images
    
    def get_path(self):
        return self.__path
    
    def get_images(self):
        return self.images
    
class DetectorDescriptor(Image):
    """
    DetectorDescriptor class provides methods to detect keypoints and compute descriptors using various algorithms such as Harris, FAST, ORB, SIFT, and AKAZE.
    Methods:
        __init__(self, path):
            Initializes the DetectorDescriptor object with the given image path.
        harris_detector(self, images=None):
            Converts each image to grayscale, applies the Harris corner detection, dilates the detected corners, and marks them in red on the original image.
            Returns a list of keypoints for each image.
        fast_detector(self, images=None):
            Returns a list of keypoints for each image.
        orb_detector(self, images=None):
            Returns a tuple containing lists of keypoints, descriptors, and processing times for each image.
        sift_detector(self, images=None):
            Returns a tuple containing lists of keypoints, descriptors, and processing times for each image.
        akaze_descriptor(self, images=None):
            Returns a tuple containing lists of keypoints, descriptors, and processing times for each image.
    """
    def __init__(self,path):
        super().__init__(path)

    def harris_detector(self, images = None):
        """
        Applies the Harris corner detection algorithm to each image in the self.images list.
        The function converts each image to grayscale, applies the Harris corner detection,
        dilates the detected corners, and marks them in red on the original image. The processed
        images are stored in the self.harris_images attribute.
        Returns:
            None
        """
        if images is None:
            images = np.copy(self.get_images())
            
        keypoints = []
        for img in images:
            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)               #harris works on float32
            dst = cv2.cornerHarris(gray, 5, 5, 0.04)
            dst = cv2.dilate(dst, None)
            
            # Extract keypoints from Harris response
            kp = np.argwhere(dst > 0.01 * dst.max())
            keypoints.append(kp.astype(np.float32))
            
        return keypoints

    def fast_detector(self, images = None):
        """
        Detects keypoints in a list of images using the FAST (Features from Accelerated Segment Test) algorithm.
        Parameters:
        images (list of numpy.ndarray, optional): List of images in which to detect keypoints. 
                                                  If None, the method will call self.get_images() to retrieve images.
        Returns:
        list of numpy.ndarray: A list where each element is an array of keypoints detected in the corresponding image.
                               Each keypoint is represented by its (x, y) coordinates.
        """
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        
        keypoints = []
        if images is None:
            images = self.get_images()
            
        for img in images:
            # find and draw the keypoints
            kp = fast.detect(img,None)
            keypoints.append(np.array([kpt.pt for kpt in kp]))
        
        return keypoints
       
        
    def orb_detector(self,images = None):
        """
        Detects keypoints and computes descriptors using the ORB (Oriented FAST and Rotated BRIEF) algorithm.
        Parameters:
        images (list of numpy.ndarray, optional): List of images to process. If None, the method will use images from self.get_images().
        Returns:
        tuple: A tuple containing:
            - keypoints (list of numpy.ndarray): List of arrays containing the coordinates of the detected keypoints for each image.
            - descriptors (list of numpy.ndarray): List of arrays containing the descriptors for each image.
            - time_t (list of float): List of times taken to process each image.
        """
        """"ORB algorithm use the FAST detector and BRIEF descriptor"""
        # Initiate ORB detector
        orb = cv2.ORB_create()
        descriptors = []
        keypoints = []
        time_t = []
        if images is None:
            images = np.copy(self.get_images())
            
        for img in images:
            # compute the key points with ORB
            start_time = time.time()
            kp, des = orb.detectAndCompute(img, None)
            end_time = time.time()
            time_t.append(end_time - start_time)
            descriptors.append(des)
            keypoints.append(np.array([kpt.pt for kpt in kp]))
        
        return keypoints, descriptors, time_t
    
    def sift_detector(self,images = None):
        """
        Detects keypoints and computes descriptors using the SIFT algorithm for a list of images.
        Parameters:
        images (list of ndarray, optional): List of images to process. If None, the method will use images from self.get_images().
        Returns:
        tuple: A tuple containing:
            - keypoints (list of ndarray): List of arrays containing the keypoints for each image.
            - descriptors (list of ndarray): List of arrays containing the descriptors for each image.
            - time_t (list of float): List of times taken to compute keypoints and descriptors for each image.
        """
        sift = cv2.SIFT_create()
        time_t = []
        keypoints = []
        descriptors = []
        if images is None:
            images = np.copy(self.get_images())
            
        for img in images:
            # compute the key points with SIFT
            start_time = time.time()
            kp, des = sift.detectAndCompute(img, None)
            end_time= time.time()
            time_t.append(end_time - start_time)
            descriptors.append(des)
            keypoints.append(np.array([kpt.pt for kpt in kp]))
               
        return keypoints ,descriptors, time_t
    
    def akaze_descriptor(self,images = None):
        """
        Computes AKAZE descriptors for a list of images.
        Parameters:
        images (list of numpy.ndarray, optional): List of images for which to compute the descriptors. 
                                                  If None, the method will use images from self.get_images().
        Returns:
        tuple: A tuple containing:
            - keypoints (list of numpy.ndarray): List of keypoints for each image.
            - descriptors (list of numpy.ndarray): List of descriptors for each image.
            - time_t (list of float): List of times taken to compute the descriptors for each image.
        """
        akaze = cv2.AKAZE_create()
        keypoints = []
        descriptors = []
        time_t = []
        if images is None:
            images = np.copy(self.get_images())
            
        for img in images:
            # compute the key points with AKAZE
            start_time = time.time()
            kp, des = akaze.detectAndCompute(img, None)
            end_time = time.time()
            time_t.append(end_time - start_time)
            keypoints.append(np.array([kpt.pt for kpt in kp]))
            descriptors.append(des)
        
        return keypoints, descriptors , time_t 

                  
class KeyPoints:
    """ 
    The KeyPoints class provides methods for manipulating keypoints in images, including rotation, scaling, and threshold calculation.
    Methods:
        __init__():
            Initializes the KeyPoints class.
        rotate_keypoints(keypoints, rotation_matrix):
        adjust_keypoints_for_scale(keypoints, scale):
                scale (float): Scaling factor.
        calculate_scaled_threshold(image_shape, scaling_factor):
        calculate_threshold(image_shape):
    """
    def __init__(self): 
        pass
    
    def rotate_keypoints(self, keypoints, rotation_matrix):
        """
        Rotates the keypoints by applying the rotation matrix to their coordinates.
        Args:
            keypoints (list): A list of keypoints to be rotated.
            rotation_matrix (np.ndarray): The rotation matrix used to rotate the keypoints.
        Returns:
            list: A list of rotated keypoints.
        """
        
        # Convert keypoints to homogeneous coordinates (Nx3 array)
        homogeneous_keypoints = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
        rotated_keypoints = homogeneous_keypoints @ rotation_matrix.T
        
        return rotated_keypoints
    
    def adjust_keypoints_for_scale(self,keypoints, scale):
        """
        Adjust keypoints for scaling using NumPy for efficiency.

        Args:
            keypoints (np.ndarray): Keypoints as a NumPy array of shape (N, 2), where N is the number of keypoints.
            scale_x (float): Scaling factor in the x direction.
            scale_y (float): Scaling factor in the y direction.

        Returns:
            np.ndarray: Adjusted keypoints as a NumPy array of shape (N, 2).
        """
        # Create a scaling matrix
        scaling_matrix = np.array([scale, scale])

        # Element-wise multiplication of keypoints with the scaling factors
        adjusted_keypoints = keypoints * scaling_matrix

        return adjusted_keypoints
       
    def calculate_scaled_threshold(self,image_shape,scaling_factor):
        """
        Calculate a scaled threshold based on the image shape and scaling factor.

        Parameters:
        image_shape (tuple): A tuple representing the shape of the image (height, width).
        scaling_factor (float): A scaling factor to adjust the threshold.

        Returns:
        float: The calculated threshold value.
        """
        if scaling_factor < 1.0:
            return 0.05 * min(image_shape[0], image_shape[1])
        else:
            return 0.005 * max(image_shape[0], image_shape[1])
    
    def calculate_threshold(self,image_shape):
        """
        Calculate a threshold value based on the shape of the image.

        The threshold is determined as 5% of the larger dimension (height or width) of the image.

        Args:
            image_shape (tuple): A tuple representing the shape of the image (height, width).

        Returns:
            float: The calculated threshold value.
        """
        return 0.05 * max(image_shape[0], image_shape[1])
      
class Matcher(KeyPoints):
    """
    Matcher class for keypoint and descriptor matching.
    Methods:
        __init__():
            Initializes the Matcher object with a BFMatcher using L2 norm and cross-check.
        calculate_repeatability(keypoints_ref, keypoints_trans, matches, threshold):
        match_keypoints(points1, points2, threshold):
    """
    def __init__(self):
        pass
    
    def calculate_repeatability(self,keypoints_ref, keypoints_trans, matches, threshold):
        """
        Calculate repeatability between two sets of keypoints.

        Args:
            keypoints_ref (np.ndarray): Keypoints in the reference image (n1, 2).
            keypoints_trans (np.ndarray): Keypoints in the transformed image (n2, 2).
            matches (list of tuples): Matches as a list of (i, j) indices.
            threshold (float): Distance threshold to consider a match valid.

        Returns:
            float: Repeatability score (0 to 1).
        """
        valid_matches = 0
        
        for i, j in matches:
            # Compute Euclidean distance between matched keypoints
            distance = np.linalg.norm(keypoints_ref[i] - keypoints_trans[j])
            
            # Check if the match is valid
            if distance <= threshold:
                valid_matches += 1
        
        # Calculate repeatability
        repeatability = valid_matches / max(len(keypoints_ref),len(keypoints_trans))
        
        return repeatability

     
    def match_keypoints(self, points1, points2, threshold):
        """
        Matches keypoints between two sets of points based on Euclidean distance.
        Args:
            points1 (ndarray): An array of shape (n1, d) representing the first set of keypoints.
            points2 (ndarray): An array of shape (n2, d) representing the second set of keypoints.
            threshold (float): The maximum allowable distance for a match.
        Returns:
            list of tuples: A list of matched keypoints indices (i, j) where i is the index in points1 and j is the index in points2.
        """
        from scipy.spatial import distance
        # Compute distances between all pairs of keypoints
        distances = distance.cdist(points1, points2, metric='euclidean')  # Shape (n1, n2)
        matches = []
        # Find the nearest neighbor for each keypoint in keypoints1
        for i in range(len(points1)):
            j = np.argmin(distances[i])  # Index of the nearest neighbor in keypoints2
            if distances[i, j] <= threshold:  # Apply distance threshold
                matches.append((i, j))
        
        return matches
    
    def match_descriptors(self, descriptors1, descriptors2):
        """
        Matches descriptors between two sets using the BFMatcher with L2 norm and cross-check.
        Args:
            descriptors1 (numpy.ndarray): The first set of descriptors.
            descriptors2 (numpy.ndarray): The second set of descriptors.
        Returns:
            list: A list of DMatch objects containing the matched descriptors.
        """
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(descriptors1, descriptors2)
        
        return matches
    
    def calculate_matching_accuracy(self, keypoints_ref, keypoints_trans, matches,threshold):
        """
        Calculate accuracy of descriptor matching.

        Args:
            keypoints_ref (np.ndarray): Keypoints in the reference image (n1, 2).
            keypoints_trans (np.ndarray): Keypoints in the transformed image (n2, 2).
            matches (list of tuples): Matches as a list of (i, j) indices.
            homography (np.ndarray): Homography matrix from reference to transformed image.
            threshold (float): Distance threshold to consider a match correct.

        Returns:
            float: Accuracy score (0 to 1).
        """
        # Extract match points
        src_pts = np.float32([keypoints_ref[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_trans[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        
        # Compute homography
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        correct_matches = 0
        
        for match in matches:
            # Retrieve the indices of the matched keypoints
            i = match.queryIdx  # Index of the keypoint in the reference image
            j = match.trainIdx  # Index of the keypoint in the transformed image
            # Project reference keypoint to transformed image using homography
            pt_ref = np.array([*keypoints_ref[i], 1])  # Homogeneous coordinates
            projected_pt = homography @ pt_ref
            projected_pt /= projected_pt[2]  # Normalize homogeneous coordinates
            
            # Compare with the actual transformed keypoint
            distance = np.linalg.norm(projected_pt[:2] - keypoints_trans[j])
            
            # Check if the match is correct
            if distance <= threshold:
                correct_matches += 1
        
        # Calculate accuracy
        accuracy = correct_matches / len(matches) if matches else 0
        
        return accuracy

    
    def calculate_positioning_error(self, kp1, kp2, matches):
        """
        Calculate the positioning error between matching keypoints in two images.
        
        :param kp1: A numpy array of shape (N, 2) for the first set of keypoints.
        :param kp2: A numpy array of shape (M, 2) for the second set of keypoints.
        :param matches: A list of tuples, where each tuple contains the indices of matching keypoints in kp1 and kp2.
        :return: The average positioning error across all matches.
        """
        errors = []
        
        # Loop over all matches
        for match in matches:
            i1, i2 = match  # Indices of matching keypoints in kp1 and kp2
            
            # Get the corresponding points from both sets
            point1 = kp1[i1]  # Point in the first image
            point2 = kp2[i2]  # Point in the second image
            
            # Calculate the Euclidean distance between the corresponding points
            error = np.linalg.norm(point1 - point2)
            errors.append(error)
        
        # Calculate the average positioning error
        average_error = np.mean(errors) if errors else 0
        return average_error
