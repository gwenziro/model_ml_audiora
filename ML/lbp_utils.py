import cv2
import numpy as np
from skimage.feature import hog

def extract_lbp_hog_features(image):
    """
    Extract combined LBP and HOG features from an image.
    """
    # Resize image to a fixed size (e.g., 100x100)
    fixed_size = (100, 100)
    image_resized = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Extract LBP features
    lbp_image = np.zeros_like(gray_image, dtype=np.uint8)
    for i in range(1, gray_image.shape[0] - 1):
        for j in range(1, gray_image.shape[1] - 1):
            center = gray_image[i, j]
            binary_string = ''.join(['1' if gray_image[i + x, j + y] >= center else '0'
                                     for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                                  (1, 1), (1, 0), (1, -1), (0, -1)]])
            lbp_image[i, j] = int(binary_string, 2)

    lbp_features = lbp_image.flatten()

    # Extract HOG features
    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # Combine LBP and HOG features
    combined_features = np.concatenate((lbp_features, hog_features))

    return combined_features
