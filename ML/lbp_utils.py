import cv2
import numpy as np
import pickle

def extract_lbp_features(image_path):
    """Extract LBP features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1  # Radius for LBP
    n_points = 8 * radius  # Number of circular points

    # Compute LBP
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Normalize the histogram
    lbp_normalized = lbp / lbp.sum()
    return lbp_normalized.flatten()

def predict_age_group(image_path, model_path='ML/models/knn_model.pkl'):
    """Predict the age group using a pre-trained KNN model."""
    lbp_features = extract_lbp_features(image_path)

    # Load the pre-trained model
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)

    # Predict age
    age_group = knn_model.predict([lbp_features])[0]
    return age_group
