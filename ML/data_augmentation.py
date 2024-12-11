import numpy as np
import cv2
import albumentations as A

def augment_image(image):
    """
    Lakukan augmentasi pada gambar input menggunakan albumentations.

    Parameter:
    - image: np.ndarray, gambar input.

    Mengembalikan:
    - augmented_image: np.ndarray, gambar hasil augmentasi.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5)
    ])
    augmented = transform(image=image)
    augmented_image = augmented['image']
    return augmented_image