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


import os  # Add this line
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from lbp_utils import extract_lbp_features

def prepare_data(data_dir):
    """
    Prepare data from the given directory, loading images and corresponding labels.
    Assumes images are organized into subdirectories by age group.
    """
    images = []
    labels = []

    # Traverse the dataset directory
    for label_dir in os.listdir(data_dir):  # This line will now work correctly
        label_path = os.path.join(data_dir, label_dir)
        
        if os.path.isdir(label_path):
            label = int(label_dir)  # Age group as the label
            
            for filename in os.listdir(label_path):
                image_path = os.path.join(label_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image = cv2.resize(image, (64, 64))  # Resize to a consistent size
                
                # Extract LBP features
                features, lbp_values = extract_lbp_features(image)
                
                images.append(features)
                labels.append(label)
                
                # Show the LBP values for each image during training
                print(f"LBP values for {filename}: {lbp_values}")
    
    return np.array(images), np.array(labels)

def test_knn_model(test_image_path, data_dir, k=5):
    """
    Test the KNN model without using the saved model file.
    It will train the model on the dataset and then test it on the provided image.
    """
    # Prepare data (training set)
    X, y = prepare_data(data_dir)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize KNN classifier with k and Manhattan distance
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Test the model (evaluate on the test set)
    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy on the test set: {accuracy * 100:.2f}%")
    
    # Read and preprocess the test image
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (64, 64))  # Resize to match training size
    test_image = test_image / 255.0  # Normalize the pixel values

    # Extract LBP features from the test image
    histogram, lbp_values = extract_lbp_features(test_image)
    
    # Display the LBP values for the test image
    print(f"LBP values for the test image: {lbp_values}")
    
    # Predict the age group using the KNN model
    predicted_label = knn.predict(histogram.reshape(1, -1))[0]
    
    # Calculate confidence (distance metric)
    distances, indices = knn.kneighbors(histogram.reshape(1, -1))
    confidence = 1 / (1 + np.mean(distances))  # Example confidence calculation
    
    # Display prediction results
    print(f"Predicted Age Group: {predicted_label}")
    print(f"Confidence: {confidence * 100:.2f}%")

# Test the model with a test image
test_image_path = r'D:\Be\BE\ML\uploads\images\Shamil.jpg'  # Replace with your actual test image path
data_dir = r'D:\Be\BE\ML\uploads\age'  # Replace with your actual dataset path

test_knn_model(test_image_path, data_dir)
