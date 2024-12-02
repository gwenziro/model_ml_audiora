import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from knn_model import prepare_data, label_mapping, detect_face
from lbp_utils import extract_lbp_features

def test_knn_model(test_image_path, data_dir):
    """
    Test the KNN model on a test image and display the top-3 matches.
    """
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

    # Evaluate the model on the test split
    print("Evaluating model...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of KNN model: {accuracy * 100:.2f}%")

    # Test the model on the input image
    print(f"Testing on image: {test_image_path}")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return

    # Detect the face in the test image
    face_region = detect_face(test_image)
    if face_region is None:
        print("Error: No face detected in the test image.")
        return

    # Extract LBP features and reduce dimensions using PCA
    feature_vector, _ = extract_lbp_features(face_region)
    feature_reduced = ipca.transform([feature_vector])

    # Predict the class and display the result
    prediction = knn.predict(feature_reduced)[0]
    suggestion_name = label_mapping[prediction]
    print(f"Predicted Age Group: {suggestion_name}")

    # Display the test image
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {suggestion_name}")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    data_dir = 'D:/Be/BE/ML/uploads/age'  # Path to your dataset directory
    test_image_path = r'D:\Be\BE\ML\uploads\images\seno.jpg'  # Path to the test image

    test_knn_model(test_image_path, data_dir)
