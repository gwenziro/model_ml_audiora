from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from knn_model import prepare_data_with_faces, label_mapping
from lbp_utils import extract_face_and_lbp_features
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def test_knn_model_with_faces(test_image_path, data_dir):
    """
    Test the KNN model on a test image and display top-3 matches.
    """
    print("Preparing dataset...")
    X, y, ipca = prepare_data_with_faces(data_dir)
    print(f"Total features extracted: {len(X)}")
    print(f"Labels distribution: {np.unique(y, return_counts=True)}")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    print("KNN model trained.")

    # Evaluate the model
    print("Evaluating model...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of KNN model: {accuracy * 100:.2f}%")

    # Test on the input image
    print(f"Testing on image: {test_image_path}")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return

    # Extract LBP features from the face in the test image
    feature = extract_face_and_lbp_features(test_image)
    if feature is None:
        print("No face detected in the test image.")
        return

    feature_reduced = ipca.transform([feature])  # Apply the same PCA as used for the dataset

    # Get the top-3 matches
    distances, indices = knn.kneighbors(feature_reduced, n_neighbors=3)
    top_matches = [(y[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    print("\nTop-3 Matches:")
    for i, (label, distance) in enumerate(top_matches):
        label_name = [key for key, value in label_mapping.items() if value == label][0]
        print(f"{i + 1}. Label: {label}, Name: {label_name}, Distance: {distance:.2f}")

# Example usage
data_dir = r"D:\Be\BE\ML\uploads\age"
test_image_path = r"D:\Be\BE\ML\uploads\images\febiola.JPEG"

test_knn_model_with_faces(test_image_path, data_dir)
