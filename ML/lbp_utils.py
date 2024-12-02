import numpy as np
from skimage.feature import local_binary_pattern
import cv2

def extract_lbp_features(image, size=(8, 8)):
    """
    Extract Local Binary Pattern (LBP) features from the given image (face region).
    Parameters:
        - image: Grayscale image of the face region.
        - size: Desired size of the output feature vector.
    Returns:
        - feature_vector: Flattened LBP feature vector.
        - lbp_image: LBP image visualization.
    """
    radius = 2  # Increased radius
    n_points = 8 * radius  # Number of circularly symmetric neighbor points
    method = 'uniform'  # LBP method

    # Compute the LBP image
    lbp = local_binary_pattern(image, n_points, radius, method=method)

    # Normalize and resize LBP to match the desired size
    lbp_resized = cv2.resize(lbp.astype(np.float32), size)
    lbp_resized = (lbp_resized - lbp_resized.min()) / (lbp_resized.max() - lbp_resized.min())
    feature_vector = lbp_resized.flatten()

    return feature_vector, lbp
