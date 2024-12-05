import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(image):
    """
    Ekstrak fitur LBP dari wajah pada gambar. Jika wajah tidak terdeteksi,
    gunakan seluruh gambar.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Langkah 1: Fokus pada wajah (hilangkan background, rambut, dll.)
        x, y, w, h = faces[0]
        face_region = gray_image[y:y+h, x:x+w]
    else:
        face_region = gray_image

    # Resize untuk dimensi yang konsisten
    face_resized = cv2.resize(face_region, (100, 100), interpolation=cv2.INTER_AREA)

    # Ekstrak fitur LBP
    lbp = local_binary_pattern(face_resized, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    return lbp_hist

def extract_hog_features(image):
    """
    Ekstrak fitur HOG dari wajah pada gambar.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    resized_image = cv2.resize(gray_image, (64, 128))
    hog_features = hog.compute(resized_image)
    return hog_features.flatten()
