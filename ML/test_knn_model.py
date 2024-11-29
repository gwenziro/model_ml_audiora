# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import numpy as np
# import pickle
# import cv2
# from lbp_utils import extract_lbp_features
# import os
# from sklearn.neighbors import KNeighborsClassifier

# def load_database(database_path):
#     images = []
#     labels = []
#     # Loop through each folder in the database path
#     for folder_name in os.listdir(database_path):
#         folder_path = os.path.join(database_path, folder_name)
        
#         # Ensure it's a directory and the folder name is a valid age label
#         if os.path.isdir(folder_path) and folder_name.isdigit():
#             label = int(folder_name)  # Folder name represents the age (age label)
#             # Loop through each image inside the folder
#             for image_name in os.listdir(folder_path):
#                 image_path = os.path.join(folder_path, image_name)
#                 # Check if the image file is valid (jpg, jpeg, png)
#                 if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     image = cv2.imread(image_path)
#                     # Check if the image was loaded successfully
#                     if image is not None:
#                         images.append(image)
#                         labels.append(label)
#     return images, labels

# def train_knn_model(database_path):
#     images, ages = load_database(database_path)
    
#     features = []
#     for image in images:
#         if image is not None:
#             # Extract LBP features from the image
#             feature_vector, _ = extract_lbp_features(image, size=(8, 8))
#             features.append(feature_vector)
    
#     # Convert the features and labels to numpy arrays
#     features = np.array(features)
#     ages = np.array(ages)

#     print(f"Number of features extracted for training: {features.shape[1]}")  # Check number of features
    
#     # Train the KNN model
#     knn_model = KNeighborsClassifier(n_neighbors=3)
#     knn_model.fit(features, ages)

#     # Save the trained model to a file
#     with open('knn_model.pkl', 'wb') as model_file:
#         pickle.dump(knn_model, model_file)

#     print("Model trained and saved.")

# def predict_and_evaluate(database_path, image_path):
#     images, true_ages = load_database(database_path)
    
#     # Extract features for all images in the dataset
#     features = []
#     for image in images:
#         if image is not None:
#             feature_vector, _ = extract_lbp_features(image, size=(8, 8))
#             features.append(feature_vector)
    
#     features = np.array(features)

#     # Load the trained KNN model
#     with open('knn_model.pkl', 'rb') as model_file:
#         knn_model = pickle.load(model_file)

#     # Predict age for each image
#     predicted_ages = knn_model.predict(features)
    
#     # Bin the ages into groups for classification
#     age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
#     true_labels = np.digitize(true_ages, bins=age_bins) - 1  # Convert true ages into bin labels
#     predicted_labels = np.digitize(predicted_ages, bins=age_bins) - 1  # Convert predicted ages into bin labels

#     # Calculate classification metrics
#     accuracy = accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
#     recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
#     f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")

#     # Optionally: Show the LBP feature vector and predicted age for a sample image
#     sample_image = cv2.imread(image_path)
#     feature_vector, _ = extract_lbp_features(sample_image, size=(8, 8))
#     print(f"Extracted Feature Vector: {feature_vector}")
#     predicted_age = knn_model.predict(feature_vector.reshape(1, -1))
#     print(f"Predicted Age: {predicted_age[0]}")

# if __name__ == "__main__":
#     # Path to the image for prediction
#     database_path = r'D:/Be/BE/ML/uploads/face_age'  # Path to the dataset
#     image_path = r'D:/Be/BE/ML/uploads/images/Shamil.jpg'  # Path to the image you want to test

#     # Evaluate model
#     predict_and_evaluate(database_path, image_path)

# test_knn_model.py

# test_knn_model.py (combined training and testing)



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
    display_suggestions(test_image, [idx for _, _, _ in top_matches], data_dir, y)

def display_suggestions(test_image, indices, data_dir, y):
    """
    Display the test image and top-3 suggestions using matplotlib.
    """
    print("\nDisplaying Test Image and Top-3 Suggestions...")
    plt.figure(figsize=(15, 5))

    # Show the test image
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(test_image, cv.COLOR_BGR2RGB))
    plt.title("Test Image")
    plt.axis('off')

    # Show the top-3 suggestions
    for i, idx in enumerate(indices):
        label = y[idx]
        label_name = [key for key, value in label_mapping.items() if value == label][0]
        label_dir = os.path.join(data_dir, label_name)
        suggestion_image_path = os.path.join(label_dir, os.listdir(label_dir)[0])  # First image in the folder

        suggestion_image = cv.imread(suggestion_image_path)
        plt.subplot(1, 4, i + 2)
        plt.imshow(cv.cvtColor(suggestion_image, cv.COLOR_BGR2RGB))
        plt.title(f"Match {i + 1}: {label_name}")
        plt.axis('off')

    plt.show()

# Example usage
data_dir = 'D:/Be/BE/ML/uploads/age'
test_image_path = 'D:/Be/BE/ML/uploads/images/Febiola.PNG'

test_knn_model(test_image_path, data_dir)
