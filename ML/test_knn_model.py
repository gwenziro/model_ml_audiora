import cv2
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from knn_model import prepare_data, label_mapping
from lbp_utils import extract_lbp_hog_features
import matplotlib.pyplot as plt

def test_knn_model(test_image_path, data_dir):
    """
    Test the KNN model on a test image using combined LBP+HOG features.
    """
    print("Loading dataset...")
    pca_model = None
    if os.path.exists("pca_model.pkl"):
        pca_model = joblib.load("pca_model.pkl")

    X, y, pca_model = prepare_data(data_dir, pca_model=pca_model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f"KNN Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print(f"Testing image: {test_image_path}")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return

    test_feature = extract_lbp_hog_features(test_image)
    test_feature = pca_model.transform([test_feature])

    distances, indices = knn.kneighbors(test_feature, n_neighbors=3)
    top_matches = []
    for i, idx in enumerate(indices[0]):
        label = y[idx]
        label_name = [name for name, value in label_mapping.items() if value == label][0]
        top_matches.append((label_name, distances[0][i]))

    # Print the top-3 matches first
    print("\nTop-3 Matches:")
    for i, (label_name, distance) in enumerate(top_matches):
        print(f"Match {i + 1}: {label_name}, Distance: {distance:.2f}")

    # Display the test image with predicted label
    display_test_image(test_image, top_matches[0][0])

def display_test_image(test_image, predicted_label):
    """
    Display the test image with its predicted label name.
    """
    print("\nDisplaying Test Image...")
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# Example usage
data_dir = 'D:/Be/BE/ML/uploads/age'  # Path to the dataset directory
test_image_path = r'D:\Be\BE\ML\uploads\images\ibuk1.jpeg'  # Path to the test image

test_knn_model(test_image_path, data_dir)
