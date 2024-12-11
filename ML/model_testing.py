import cv2
import numpy as np
import joblib
import os
from feature_extraction import extract_features
from config import label_mapping

def predict_age(test_image_path):
    # Dapatkan direktori skrip ini
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Bangun path absolut ke model dan objek lainnya
    knn_model_path = os.path.join(script_dir, 'knn_model.pkl')
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    svd_path = os.path.join(script_dir, 'svd.pkl')

    # Memastikan file model ada
    if not os.path.exists(knn_model_path):
        raise FileNotFoundError(f"File model tidak ditemukan pada {knn_model_path}")

    # Muat model dan objek lainnya
    knn = joblib.load(knn_model_path)
    scaler = joblib.load(scaler_path)
    svd = joblib.load(svd_path)

    # Muat gambar uji
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return

    # Ekstrak fitur
    combined_features = extract_features(test_image)

    # Normalisasi dan reduksi dimensi
    test_feature_normalized = scaler.transform([combined_features])
    test_feature_reduced = svd.transform(test_feature_normalized)

    # Prediksi
    predicted_label = knn.predict(test_feature_reduced)
    predicted_label_name = [k for k, v in label_mapping.items() if v == predicted_label[0]][0]

    return predicted_label_name