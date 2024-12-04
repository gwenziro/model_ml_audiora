import os
import cv2
import numpy as np
from sklearn.decomposition import IncrementalPCA
from lbp_utils import extract_lbp_hog_features

label_mapping = {
    '1-10 anak': 0,
    '11-20 remaja': 1,
    '21-30 transisi': 2,
    '31-40 masa matang': 3,
    '41-50 dewasa': 4,
    '51-60 usia pertengahan': 5,
    '61-70 tua': 6,
    '71-80 lanjut usia': 7,
    '81-90 lanjut usia tua': 8,
}

def prepare_data(data_dir, pca_model=None):
    """
    Load the dataset, extract LBP+HOG features, and apply PCA for dimensionality reduction.
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

                # Skip images without detectable faces
                if image is not None:
                    feature = extract_lbp_hog_features(image)
                    features.append(feature)
                    labels.append(label)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)

    # Apply PCA if provided, or fit a new one
    if pca_model is None:
        pca_model = IncrementalPCA(n_components=200)  # Adjust components based on the dataset
        features = pca_model.fit_transform(features)
    else:
        features = pca_model.transform(features)

    return features, labels, pca_model
