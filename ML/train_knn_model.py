import os
import joblib
from data_preparation import prepare_data
from model_training import train_knn

if __name__ == "__main__":
    data_dir = r"D:\\PBL\\backend-febi\\BE\\ML\\uploads\\age"  # Sesuaikan dengan path dataset Anda

    # Periksa apakah data yang diproses sudah ada
    if os.path.exists('features_labels.pkl'):
        print("=== Memuat Data yang Telah Diproses ===")
        X_reduced, labels_resampled = joblib.load('features_labels.pkl')
        print("=== Memuat Data Selesai ===")
    else:
        print("=== Mulai Persiapan Data ===")
        X_reduced, labels_resampled = prepare_data(data_dir)
        print("=== Persiapan Data Selesai ===")

    print("=== Mulai Pelatihan Model ===")
    train_knn(X_reduced, labels_resampled)
    print("=== Pelatihan Model Selesai ===")