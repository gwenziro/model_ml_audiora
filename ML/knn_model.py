import os
import numpy as np
from sklearn.decomposition import PCA
from lbp_utils import extract_lbp_hog_features
import joblib
import cv2

label_mapping = {
    '1-10 anak': 0,
    '11 - 20 remaja': 1,
    '21-30 transisi': 2,
    '31-40 masa matang': 3,
    '41-50 dewasa': 4,
    '51-60 usia pertengahan': 5,
    '61-70 tua': 6,
    '71-80 lanjut usia': 7,
    '81-90 lanjut usia tua': 8
}

def prepare_data(data_dir, pca_model=None):
    """
    Load the dataset, extract features, and apply PCA for dimensionality reduction.
    """
    features = []
    labels = []

    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)

        if os.path.isdir(label_path):
            label = label_mapping.get(label_dir, -1)
            if label == -1:
                print(f"Warning: Unmapped label directory: {label_dir}")
                continue
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    feature = extract_lbp_hog_features(image)
                    features.append(feature)
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    if pca_model is None:
        pca_model = PCA(n_components=100)  # Adjust based on your feature space
        features = pca_model.fit_transform(features)
        # Save the PCA model for later use
        joblib.dump(pca_model, 'pca_model.pkl')
    else:
        features = pca_model.transform(features)

    return features, labels, pca_model
