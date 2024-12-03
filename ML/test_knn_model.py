from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from knn_model import prepare_data, label_mapping
from lbp_utils import extract_lbp_features
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def test_knn_model(test_image_path, data_dir):
    """
    Test the KNN model on a test image, compare LBP features, and display the top-3 matches.
    """
    print("Preparing dataset...")
    X, y, ipca = prepare_data(data_dir)
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
    test_image = cv.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return

    # Extract LBP features from the test image
    feature, _ = extract_lbp_features(test_image)
    feature_reduced = ipca.transform([feature])  # Apply the same PCA as used for the dataset

    # Get the top-10 neighbors
    distances, indices = knn.kneighbors(feature_reduced, n_neighbors=10)

    # Ensure unique labels in top-3 matches
    top_matches = []
    seen_labels = set()
    for i, idx in enumerate(indices[0]):
        suggestion_label = y[idx]
        if suggestion_label not in seen_labels:
            suggestion_name = [key for key, value in label_mapping.items() if value == suggestion_label][0]
            top_matches.append((suggestion_label, suggestion_name, distances[0][i]))
            seen_labels.add(suggestion_label)
        if len(top_matches) == 3:
            break

    print("\nTop-3 Matches:")
    for i, (label, name, distance) in enumerate(top_matches):
        print(f"{i + 1}. Label: {label}, Name: {name}, Distance: {distance:.2f}")

    # Display the test image and top-3 suggestions
    display_suggestions(test_image, top_matches, data_dir)

def display_suggestions(test_image, top_matches, data_dir):
    """
    Display the test image and the corresponding images of the top-3 matches.
    """
    print("\nDisplaying Test Image and Top-3 Suggestions...")
    plt.figure(figsize=(15, 5))

    # Show the test image
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(test_image, cv.COLOR_BGR2RGB))
    plt.title("Test Image")
    plt.axis('off')

    # Iterate through the top-3 matches
    for i, (label, label_name, distance) in enumerate(top_matches):
        label_dir = os.path.join(data_dir, label_name)

        # Fetch the first image in the folder for this label
        if os.path.exists(label_dir) and os.listdir(label_dir):
            suggestion_images = os.listdir(label_dir)
            suggestion_image_path = os.path.join(label_dir, suggestion_images[0])  # Take the first image
            suggestion_image = cv.imread(suggestion_image_path)

            if suggestion_image is not None:
                plt.subplot(1, 4, i + 2)  # i + 2 ensures correct position after the test image
                plt.imshow(cv.cvtColor(suggestion_image, cv.COLOR_BGR2RGB))
                plt.title(f"Match {i + 1}: {label_name}\nDist: {distance:.2f}")
                plt.axis('off')
            else:
                print(f"Error: Could not load image for label {label_name}.")
        else:
            print(f"Error: No images found for label {label_name}.")

    plt.show()

# Example usage
data_dir = 'D:/Be/BE/ML/uploads/age'  # Path to the dataset directory
test_image_path = r'D:\Be\BE\ML\uploads\images\febiola.JPEG'  # Path to the test image

test_knn_model(test_image_path, data_dir)