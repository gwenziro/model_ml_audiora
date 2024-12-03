import numpy as np
import cv2

def extract_lbp_features(image):
    """
    Extract LBP (Local Binary Pattern) features from an image.
    """
    # Resize the image to reduce feature dimensions
    image_resized = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)

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

    # Flatten the LBP image to a feature vector
    lbp_features = lbp_image.flatten()

    return lbp_features, lbp_image