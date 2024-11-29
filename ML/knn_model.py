# import cv2
# import numpy as np
# from skimage.feature import local_binary_pattern
# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from lbp_utils import extract_lbp_features
# from test_knn_model import load_database

# def train_knn_model(database_path):
#     images, ages = load_database(database_path) 
    
#     features = []
#     for image in images:
#         if image is not None:
#             feature_vector, _ = extract_lbp_features(image, size=(8, 8))  # Ensure consistent size
#             features.append(feature_vector)
    
#     features = np.array(features)
#     ages = np.array(ages)

#     print(f"Number of features extracted for training: {features.shape[1]}")  # Check features
    
#     # Train the KNN model
#     knn_model = KNeighborsClassifier(n_neighbors=7)
#     knn_model.fit(features, ages)

#     # Save the trained model
#     with open('knn_model.pkl', 'wb') as model_file:
#         pickle.dump(knn_model, model_file)

#     print("Model trained and saved.")

# if __name__ == "__main__":
#     database_path = r'D:\Be\BE\ML\uploads\face_age'  # Update this path
#     train_knn_model(database_path)

# knn_model.py


import os
import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from lbp_utils import extract_lbp_features

# Label mapping
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

def prepare_data(data_dir):
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
                    feature, _ = extract_lbp_features(image)
                    features.append(feature)
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Apply PCA for dimensionality reduction
    ipca = IncrementalPCA(n_components=100)
    X_reduced = ipca.fit_transform(features)

    return X_reduced, labels, ipca
