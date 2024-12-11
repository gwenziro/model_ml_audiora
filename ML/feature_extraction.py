import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from config import hog_weight, lbp_weight

def detect_face_region(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_region = gray_image[y:y+h, x:x+w]
    else:
        face_region = gray_image  # Jika wajah tidak terdeteksi, gunakan seluruh gambar
    return face_region

def extract_lbp_features(face_region):
    """
    Ekstrak fitur LBP dari wajah pada gambar.
    """
    face_resized = cv2.resize(face_region, (128, 128), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(face_resized, P=24, R=8, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, density=True)
    return lbp_hist

def extract_hog_features(face_region):
    """
    Ekstrak fitur HOG dari wajah pada gambar.
    """
    resized_image = cv2.resize(face_region, (64, 128))
    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_features = hog.compute(resized_image)
    return hog_features.flatten()

def extract_features(image):
    """
    Gabungkan fitur LBP dan HOG dengan bobot masing-masing.
    """
    face_region = detect_face_region(image)
    lbp_features = extract_lbp_features(face_region)
    hog_features = extract_hog_features(face_region)
    combined_features = np.hstack((lbp_features * lbp_weight, hog_features * hog_weight))
    return combined_features