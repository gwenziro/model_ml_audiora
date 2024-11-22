import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate and setup Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset from Kaggle
api.dataset_download_files('frabbisw/facial-age', path='ML/uploads/', unzip=True)

# Define path to the 'age' folder that contains the 99 subfolders
dataset_path = 'ML/uploads/age/'

# Function to extract LBP features from images
def extract_lbp_features(image_path):
    """Extract LBP features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image from {image_path}")

    radius = 1  # Radius for LBP
    n_points = 8 * radius  # Number of circular points

    # Compute LBP
    lbp = cv2.localBinaryPattern(image, n_points, radius, method='uniform')

    # Compute the histogram of LBP values
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))

    # Normalize the histogram
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= lbp_hist.sum()
    return lbp_hist

# Collect images and labels
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

# Function to predict the age group of a new image
def predict_age(image_path, model_path='ML/models/knn_model.pkl'):
    """Predict the age group using the pre-trained KNN model."""
    try:
        lbp_features = extract_lbp_features(image_path)
    except Exception as e:
        raise ValueError(f"Error extracting LBP features: {e}")

    # Load the pre-trained KNN model
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)

    # Predict the age group
    predicted_age = knn_model.predict([lbp_features])
    return predicted_age[0]

# Path to the custom image
image_path = r'D:\Be\BE\ML\uploads\Febiola lidya Sianturi - 3x4.PNG'  # Your image path

try:
    predicted_age = predict_age(image_path)
    print(f"Predicted age group: {predicted_age}")
except Exception as e:
    print(f"Error: {e}")
