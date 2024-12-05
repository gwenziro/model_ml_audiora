import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lbp_utils import extract_lbp_features, extract_hog_features
import cv2

# Pemetaan label
label_mapping = {
    '1-10 anak': 0,
    '11-20 remaja': 1,
    '21-30 transisi': 2,
    '31-40 masa matang': 3,
    '41-50 dewasa': 4,
    '51-60 usia pertengahan': 5,
    '61-70 tua': 6,
    '71-80 lanjut usia': 7,
    '81-90 lanjut usia tua': 8
}

def prepare_data(data_dir):
    """
    Load dataset, ekstrak fitur LBP + HOG, normalisasi, dan reduksi dimensi.
    """
    features = []
    labels = []

    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            label = label_mapping.get(label_dir, -1)
            if label == -1:
                print(f"Warning: Unmapped label directory: {label_dir}")
                continue

            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    lbp_features = extract_lbp_features(image)
                    hog_features = extract_hog_features(image)

                    # Gabungkan fitur dengan bobot
                    combined_features = np.hstack((lbp_features * 0.3, hog_features * 0.7))
                    features.append(combined_features)
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Normalisasi fitur
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(features)

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=min(len(features[0]), 50))
    X_reduced = svd.fit_transform(X_normalized)

    return X_reduced, labels, scaler, svd

def train_knn(X, y, k=7):
    """
    Latih model KNN dengan jarak Manhattan.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN dengan jarak Manhattan
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(X_train, y_train)

    # Evaluasi model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracy * 100:.2f}%")

    return knn, X_train, X_test, y_train, y_test
