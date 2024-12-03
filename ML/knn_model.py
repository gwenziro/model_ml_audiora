
import os
import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from lbp_utils import extract_face_and_lbp_features

# Label mapping
label_mapping = {
    '1-10 anak': 0,
    '11-20 remaja': 1,
    '21-30 transisi': 2,
    '31-40 masa matang': 3,
    '41-50 dewasa': 4,
    '51-60 usia pertengahan': 5,
    '61-70 tua': 6,
    '71-80 lanjut usia': 7,
    '81-90 lanjut usia tua': 8
}

def prepare_data_with_faces(data_dir):
    """
    Load the dataset, extract face-specific LBP features, and apply PCA for dimensionality reduction.
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
                    feature = extract_face_and_lbp_features(image)
                    if feature is not None:  # Only append if a face was detected
                        features.append(feature)
                        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Apply PCA for dimensionality reduction
    ipca = IncrementalPCA(n_components=100)
    X_reduced = ipca.fit_transform(features)

    return X_reduced, labels, ipca

