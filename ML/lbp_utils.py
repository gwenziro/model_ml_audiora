import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Define path to the 'age' folder that contains the 99 subfolders
dataset_path = r'D:/BE/BE/ML/uploads/face_age/'

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

# Collect images and labels
X_train = []
y_train = []

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

print(f"Collected {len(X_train)} samples.")

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Ensure X_train is a 2D array
if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)

# Define absolute path for saving the model
model_path = r'D:\Be\BE\ML\models\knn_model.pkl'  # Direct absolute path
model_dir = os.path.dirname(model_path)  # Extract directory from the full path

# Check if the directory exists and create it if it doesn't
if not os.path.exists(model_dir):
    print(f"Directory {model_dir} does not exist. Creating it now...")
    os.makedirs(model_dir)
else:
    print(f"Directory {model_dir} already exists.")

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save the trained model
print(f"Saving the model to {model_path}...")
with open(model_path, 'wb') as f:
    pickle.dump(knn, f)

print(f"Model trained and saved successfully at {model_path}!")
