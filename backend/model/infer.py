import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load model
model = load_model("backend/outputs/models/final_model.h5")

# Map labels
label_map = {0:'A+', 1:'A-', 2:'B+', 3:'B-', 4:'AB+', 5:'AB-', 6:'O+', 7:'O-'}

def predict_blood_group(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to 3 channels
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # batch dimension
    pred = model.predict(img)
    label = np.argmax(pred)
    return label_map[label]

if __name__ == "__main__":
    path = input("Enter image path: ")
    result = predict_blood_group(path)
    print("Predicted Blood Group:", result)
