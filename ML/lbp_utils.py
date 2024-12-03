import numpy as np
from skimage.feature import local_binary_pattern, hog
import cv2

def sharpen_image(image):
    """
    Sharpen the given image using a sharpening kernel.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def extract_lbp_features(image, size=(8, 8)):
    """
    Extract Local Binary Pattern (LBP) features.
    """
    radius = 2
    n_points = 8 * radius
    method = 'uniform'
    
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    lbp_resized = cv2.resize(lbp.astype(np.float32), size)
    return lbp_resized.flatten()

def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    """
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features

def extract_combined_features(image, lbp_size=(8, 8)):
    """
    Combine LBP and HOG features for robust texture representation.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = sharpen_image(image)
    
    # Extract LBP and HOG features
    lbp_features = extract_lbp_features(image, size=lbp_size)
    hog_features = extract_hog_features(image)
    
    # Combine features
    return np.concatenate([lbp_features, hog_features])
