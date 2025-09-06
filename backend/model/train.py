import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE_DIR, "outputs", "labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(LABELS_CSV)
    X, y = [], []

    for _, row in df.iterrows():
        file_path = os.path.join(BASE_DIR, row["file"])  # path to .npy
        arr = np.load(file_path)  # load fingerprint array
        X.append(arr)
        y.append(row["label_idx"])

    X = np.array(X) / 255.0
    X = X.reshape(-1, 128, 128, 1)  # grayscale
    num_classes = len(df["label"].unique())
    y = to_categorical(y, num_classes=num_classes)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_model(input_shape=(128,128,1), num_classes=8):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model():
    X_train, X_val, y_train, y_val = load_data()
    num_classes = y_train.shape[1]

    model = build_model(num_classes=num_classes)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10, batch_size=32)

    model_path = os.path.join(MODEL_DIR, "bloodprint_cnn.h5")
    model.save(model_path)
    print(f"✅ Model trained and saved at {model_path}")

if __name__ == "__main__":
    train_model()
