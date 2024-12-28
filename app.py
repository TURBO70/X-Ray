import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, render_template, url_for
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Directory for uploaded images
UPLOAD_FOLDER = './static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained models
dt_model = joblib.load('dt_model.pkl')
log_reg_model = joblib.load('log_reg_model.pkl')

def preprocess_image(image_path, img_size=(128, 128)):
    """Preprocess the image for model prediction."""
    img = load_img(image_path, target_size=img_size, color_mode='rgb')
    img_array = img_to_array(img) / 255.0
    img_array_flat = img_array.reshape(1, -1)
    return img_array_flat

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions."""
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    # Save the image to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Predict with each model
    dt_pred = dt_model.predict(img_array)
    log_reg_pred = log_reg_model.predict(img_array)

    # Map the predictions to class labels
    prediction = {
        'Decision Tree': 'Pneumonia' if dt_pred[0] == 1 else 'Normal',
        'Logistic Regression': 'Pneumonia' if log_reg_pred[0] == 1 else 'Normal'
    }

    # Pass predictions and image path to the template
    return render_template('result.html', prediction=prediction, image_url=url_for('static', filename=f'uploads/{file.filename}'))

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=5000)
