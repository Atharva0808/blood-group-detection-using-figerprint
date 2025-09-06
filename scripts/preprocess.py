import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (bloodprint)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
IMG_SIZE = (128, 128)

def preprocess_and_save():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    for label in os.listdir(RAW_DATA_DIR):
        raw_class_dir = os.path.join(RAW_DATA_DIR, label)
        processed_class_dir = os.path.join(PROCESSED_DATA_DIR, label)

        if not os.path.exists(processed_class_dir):
            os.makedirs(processed_class_dir)

        for img_name in os.listdir(raw_class_dir):
            img_path = os.path.join(raw_class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0

            save_name = img_name.split(".")[0] + ".npy"
            save_path = os.path.join(processed_class_dir, save_name)
            np.save(save_path, img)

    print("✅ Preprocessing complete. Files saved in:", PROCESSED_DATA_DIR)

if __name__ == "__main__":
    preprocess_and_save()
