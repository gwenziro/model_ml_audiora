import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from kaggle.api.kaggle_api_extended import KaggleApi
from skimage.feature import local_binary_pattern


# Authenticate and setup Kaggle API
api = KaggleApi()
api.authenticate()

# Define path to the 'age' folder that contains the 99 subfolders
dataset_path = (r'D:/BE/BE/ML/uploads/face_age/')

# Define path to your uploaded image
uploaded_image_path = r'D:\Be\BE\ML\uploads\images\Febiola.PNG'  # Update this path

# Function to extract LBP features from images
def extract_lbp_features(image_path):
    """Extract LBP features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image from {image_path}")

    radius = 1  # Radius for LBP
    n_points = 8 * radius  # Number of circular points

    # Compute LBP using skimage
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    # Compute the histogram of LBP values
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))

    # Normalize the histogram
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()
    return lbp_hist

# Collect images and labels for training
X_train = []
y_train = []

# Iterate through all folders (age groups)
for label in os.listdir(dataset_path):
    age_folder = os.path.join(dataset_path, label)
    if os.path.isdir(age_folder):
        for image_name in os.listdir(age_folder):
            image_path = os.path.join(age_folder, image_name)
            if image_path.lower().endswith('.png'):
                try:
                    lbp_features = extract_lbp_features(image_path)
                    X_train.append(lbp_features)
                    y_train.append(int(label))  # Use the folder name as the label (age)
                except Exception as e:
                    print(f"Skipping {image_path} due to error: {e}")

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save the trained model
with open('ML/models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Model trained and saved successfully!")

# Now, extract features from your uploaded image
try:
    uploaded_image_features = extract_lbp_features(uploaded_image_path)
    
    # Use the trained model to predict the age group of the uploaded image
    predicted_age_group = knn.predict([uploaded_image_features])
    print(f"The predicted age group for the uploaded image is: {predicted_age_group[0]}")
    
except Exception as e:
    print(f"Error processing the uploaded image: {e}")
