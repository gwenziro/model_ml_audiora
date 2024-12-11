import os
import numpy as np
import cv2
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from data_augmentation import augment_image
from feature_extraction import extract_features
from config import label_mapping
import joblib
from tqdm import tqdm

def prepare_data(data_dir):
    features = []
    labels = []

    # List semua direktori label
    label_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for label_dir in tqdm(label_dirs, desc="Memproses Label"):
        label_path = os.path.join(data_dir, label_dir)
        label = label_mapping.get(label_dir, -1)
        if label == -1:
            print(f"Warning: Unmapped label directory: {label_dir}")
            continue

        image_files = os.listdir(label_path)

        for image_file in tqdm(image_files, desc=f"Memproses {label_dir}", leave=False):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                # Ekstraksi fitur dari gambar asli
                combined_features = extract_features(image)
                features.append(combined_features)
                labels.append(label)

                # Augmentasi gambar
                augmented_image = augment_image(image)
                combined_features_aug = extract_features(augmented_image)
                features.append(combined_features_aug)
                labels.append(label)
            else:
                print(f"Warning: Unable to read image {image_path}")

    features = np.array(features)
    labels = np.array(labels)

    # Menyeimbangkan dataset
    ros = RandomOverSampler(random_state=42)
    features_resampled, labels_resampled = ros.fit_resample(features, labels)

    # Normalisasi fitur
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(features_resampled)

    # Reduksi dimensi
    svd = TruncatedSVD(n_components=min(len(features_resampled[0]), 50), random_state=42)
    X_reduced = svd.fit_transform(X_normalized)

    # Simpan fitur dan label jika diperlukan
    joblib.dump((X_reduced, labels_resampled), 'features_labels.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(svd, 'svd.pkl')

    print("Data preparation is complete.")

    return X_reduced, labels_resampled