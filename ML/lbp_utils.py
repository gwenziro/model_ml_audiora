# import cv2
# import numpy as np
# from skimage.feature import local_binary_pattern


# def extract_lbp_features(image, size=(8, 8)):
#     # Convert image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Resize the image for consistent feature extraction
#     resized_image = cv2.resize(gray_image, size)
    
#     # Parameters for LBP (Local Binary Pattern)
#     radius = 1
#     n_points = 8 * radius  # Number of points to compare for the LBP
    
#     # Apply Local Binary Pattern
#     lbp_image = local_binary_pattern(resized_image, n_points, radius, method='uniform
#     lbp = local_binary_pattern(resized_image, n_points, radius, method='uniform')
    
#     # Calculate the histogram of LBP features
#     lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    
#     # Normalize the histogram
#     lbp_hist = lbp_hist.astype("float32")
#     lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize to avoid division by zero
    
#     # Identify dominant pattern
#     dominant_pattern_index = np.argmax(lbp_hist)
#     print("LBP Values (Histogram):", lbp_hist)
#     print("Dominant Pattern Index:", dominant_pattern_index)
#     print("Dominant Pattern Frequency:", lbp_hist[dominant_pattern_index])
    
#     return lbp_hist, resized_image

# lbp_utils.py

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.decomposition import PCA

# LBP parameters
radius = 1
n_points = 8 * radius

def extract_lbp_features(image):
    """
    Extract LBP (Local Binary Pattern) features from an image and return the normalized histogram.
    Additionally, we include HOG features for better prediction.
    """
    # Convert image to grayscale if it is not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP (Local Binary Pattern) for the image
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Calculate the histogram of the LBP image
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalize the histogram
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # To avoid division by zero

    # Extract HOG features (Histogram of Oriented Gradients)
    hog_features = hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    
    # Combine LBP and HOG features
    combined_features = np.concatenate((lbp_hist, hog_features))
    
    # Optional: Reduce dimensionality with PCA (if necessary)
    if combined_features.shape[0] > 1:  # Only apply PCA if there are enough features
        n_components = min(50, combined_features.shape[0])  # Make sure n_components is not larger than the number of features
        pca = PCA(n_components=n_components)
        combined_features_reduced = pca.fit_transform(combined_features.reshape(1, -1))
        return combined_features_reduced.flatten(), lbp_hist  # Return both reduced features and original LBP values
    else:
        return combined_features, lbp_hist  # Return unmodified features and original LBP values
