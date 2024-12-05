import cv2
import matplotlib.pyplot as plt
from knn_model import prepare_data, train_knn, label_mapping
from lbp_utils import extract_lbp_features, extract_hog_features
import numpy as np

def test_knn_model(test_image_path, data_dir):
    """
    Uji model KNN pada gambar uji dan tampilkan Top-3 prediksi.
    """
    # Siapkan data dan latih model
    X, y, scaler, svd = prepare_data(data_dir)
    knn, _, _, _, _ = train_knn(X, y, k=11)

    # Muat gambar uji
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print("Error: Test image not found.")
        return

    # Ekstrak fitur LBP + HOG dari gambar uji
    lbp_features = extract_lbp_features(test_image)
    hog_features = extract_hog_features(test_image)

    # Gabungkan fitur dengan bobot dan normalisasi
    combined_features = np.hstack((lbp_features * 0.3, hog_features * 0.7))
    test_feature_normalized = scaler.transform([combined_features])
    test_feature_reduced = svd.transform(test_feature_normalized)

    # Prediksi Top-3
    distances, indices = knn.kneighbors(test_feature_reduced, n_neighbors=3)
    top_matches = []
    for i, idx in enumerate(indices[0]):
        label = y[idx]
        label_name = [k for k, v in label_mapping.items() if v == label][0]
        top_matches.append((label_name, distances[0][i]))

    # Tampilkan hasil prediksi
    print("\nTop-3 Matches:")
    for rank, (name, dist) in enumerate(top_matches, 1):
        print(f"{rank}. Label: {name}, Distance: {dist:.2f}")

    # Tampilkan gambar uji dengan prediksi terdekat
    display_test_image_with_label(test_image, top_matches[0][0])

def display_test_image_with_label(test_image, label_name):
    """
    Tampilkan gambar uji dengan label prediksi terdekat.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Closest Match: {label_name}", fontsize=14, color="blue")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    data_dir = r"D:\Be\BE\ML\uploads\age"  # Ganti dengan path dataset Anda
    test_image_path = r"D:\Be\BE\ML\uploads\images\seno.jpg"  # Ganti dengan path gambar uji Anda
    test_knn_model(test_image_path, data_dir)
