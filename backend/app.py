from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sqlite3
import os
from datetime import datetime

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Connect to SQLite database (creates db.sqlite3 if it doesn't exist)
conn = sqlite3.connect('db.sqlite3', check_same_thread=False)
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    predicted_label TEXT,
    confirmed_label TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()


app = Flask(__name__)
model = load_model('backend/outputs/models/final_model.h5')
label_map = {0:'A+', 1:'A-', 2:'B+', 3:'B-', 4:'AB+', 5:'AB-', 6:'O+', 7:'O-'}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file with a unique name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Preprocess the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (224, 224))  # match model input size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict the blood group
            pred = model.predict(img)
            label = np.argmax(pred)
            prediction = label_map[label]

            # Save to database
            c.execute('INSERT INTO fingerprints (image_path, predicted_label) VALUES (?, ?)',
                      (file_path, prediction))
            conn.commit()

    return render_template('index.html', prediction=prediction)
if __name__ == '__main__':
    print("Starting Flask app...")   # Add this line
    app.run(debug=True)
