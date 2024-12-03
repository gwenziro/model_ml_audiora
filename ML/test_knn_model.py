import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from knn_model import prepare_data, detect_face, label_mapping
from lbp_utils import extract_combined_features
import cv2

def test_knn_model(test_image_path, data_dir):
    print("Preparing dataset...")
    X, y, ipca = prepare_data(data_dir)
    print(f"Total features extracted: {len(X)}")
    print(f"Labels distribution: {np.unique(y, return_counts=True)}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    print("KNN model trained.")
    
    print("Evaluating model...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of KNN model: {accuracy * 100:.2f}%")
    
    print(f"Testing on image: {test_image_path}")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return
    
    face_region = detect_face(test_image)
    if face_region is None:
        print("Error: No face detected in the test image.")
        return
    
    features = extract_combined_features(face_region)
    features_reduced = ipca.transform([features])
    prediction = knn.predict(features_reduced)[0]
    print(f"Predicted Age Group: {label_mapping[prediction]}")

# Example usage
if __name__ == "__main__":
    data_dir = 'D:/Be/BE/ML/uploads/age'
    test_image_path = r'D:\Be\BE\ML\uploads\images\seno.jpg'
    test_knn_model(test_image_path, data_dir)
