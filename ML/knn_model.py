import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import pickle
from sklearn.neighbors import KNeighborsClassifier
from lbp_utils import extract_lbp_features
from test_knn_model import load_database

def train_knn_model(database_path):
    images, ages = load_database(database_path)  # Load your images and corresponding ages
    
    features = []
    for image in images:
        if image is not None:
            feature_vector, _ = extract_lbp_features(image, size=(8, 8))  # Ensure consistent size
            features.append(feature_vector)
    
    features = np.array(features)
    ages = np.array(ages)

    print(f"Number of features extracted for training: {features.shape[1]}")  # Check features
    
    # Train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(features, ages)

    # Save the trained model
    with open('knn_model.pkl', 'wb') as model_file:
        pickle.dump(knn_model, model_file)

    print("Model trained and saved.")

if __name__ == "__main__":
    database_path = r'D:\Be\BE\ML\uploads\face_age'  # Update this path
    train_knn_model(database_path)