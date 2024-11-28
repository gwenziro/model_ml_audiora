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

import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from lbp_utils import extract_lbp_features
from joblib import dump

def prepare_data(data_dir):
    """
    Prepare data from the given directory, loading images and corresponding labels.
    Assumes images are organized into subdirectories by age group.
    """
    images = []
    labels = []

    # Traverse the dataset directory
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        
        if os.path.isdir(label_path):
            label = int(label_dir)  # Age group as the label
            
            for filename in os.listdir(label_path):
                image_path = os.path.join(label_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image = cv2.resize(image, (64, 64))  # Resize to a consistent size
                
                # Extract features
                features = extract_lbp_features(image)
                
                images.append(features)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def train_knn_model(data_dir, k=5):
    """
    Train a KNN classifier on the dataset located at data_dir.
    """
    # Prepare data
    X, y = prepare_data(data_dir)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize KNN classifier with tuned k
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Evaluate on test set
    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model to a file
    dump(knn, 'knn_model.joblib')

# Train the model
train_knn_model(r'D:\Be\BE\ML\uploads\face_age')  # Replace with your actual dataset path
