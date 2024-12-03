import cv2
import numpy as np

def detect_face(image):
    """
    Detect a face in the image using OpenCV's Haar Cascade.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None  # No face detected
    # Return the first face found
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def extract_lbp_features(image):
    """
    Extract LBP features from a grayscale image.
    """
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            binary_string = ''.join(['1' if image[i + x, j + y] >= center else '0'
                                     for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                                  (1, 1), (1, 0), (1, -1), (0, -1)]])
            lbp_image[i, j] = int(binary_string, 2)
    return lbp_image.flatten()

def extract_face_and_lbp_features(image):
    """
    Detect a face and extract its LBP features.
    """
    face = detect_face(image)
    if face is None:
        return None  # No face detected
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face_resized = cv2.resize(gray_face, (100, 100), interpolation=cv2.INTER_AREA)
    return extract_lbp_features(gray_face_resized)
