import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp_features(image, size=(8, 8)):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image for consistent feature extraction
    resized_image = cv2.resize(gray_image, size)
    
    # Parameters for LBP (Local Binary Pattern)
    radius = 1
    n_points = 8 * radius  # Number of points to compare for the LBP
    
    # Apply Local Binary Pattern
    lbp = local_binary_pattern(resized_image, n_points, radius, method='uniform')
    
    # Calculate the histogram of LBP features
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalize the histogram
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize to avoid division by zero
    
    # Identify dominant pattern
    dominant_pattern_index = np.argmax(lbp_hist)
    print("LBP Values (Histogram):", lbp_hist)
    print("Dominant Pattern Index:", dominant_pattern_index)
    print("Dominant Pattern Frequency:", lbp_hist[dominant_pattern_index])
    
    return lbp_hist, resized_image
