import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lbp_utils import extract_lbp_features

# Define label mapping
label_mapping = {
    0: '1-10 anak',
    1: '11 - 20 remaja',
    2: '21-30 transisi',
    3: '31-40 masa matang',
    4: '41-50 dewasa',
    5: '51-60 usia pertengahan',
    6: '61-70 tua',
    7: '71-80 lanjut usia',
    8: '81-90 lanjut usia tua'
}

def detect_face(image):
    """
    Detect the face in the image and return the region of interest (ROI) as a grayscale image.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face_region = gray[y:y+h, x:x+w]
    
    return face_region

def prepare_data(data_dir):
    """
    Prepare the dataset, extract LBP features, balance the dataset, and apply PCA.
    """
    print("Loading images and extracting features...")
    
    images = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        
        if os.path.isdir(label_path):
            label = list(label_mapping.values()).index(label_dir)  # Map folder name to label
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path)
                
                if image is not None:
                    face_region = detect_face(image)
                    if face_region is not None:
                        feature_vector, _ = extract_lbp_features(face_region)
                        images.append(feature_vector)
                        labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    images_balanced, labels_balanced = smote.fit_resample(images, labels)

    # Apply PCA for dimensionality reduction
    ipca = PCA(n_components=50)  # Adjust the number of components as needed
    images_pca = ipca.fit_transform(images_balanced)
    
    return images_pca, labels_balanced, ipca
