from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Class names from the model
class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 
               'melanoma', 'nevus', 'pigmented benign keratosis', 
               'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

# Load the model
model_path = 'model.h5'
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        logger.info("Model loaded successfully")
    else:
        raise FileNotFoundError("Model file not found. Please ensure model.h5 exists in the root directory.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def validate_image(img):
    """Validate image quality and characteristics."""
    try:
        # Check image size
        width, height = img.size
        if width < 100 or height < 100:
            return False, "Image is too small. Minimum size is 100x100 pixels."
        
        return True, "Image validation successful"
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"

def preprocess_image(img):
    try:
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            logger.info("Converting RGBA image to RGB")
            img = img.convert('RGB')
        
        # Validate image
        is_valid, message = validate_image(img)
        if not is_valid:
            raise ValueError(message)
        
        # Resize image to match model's expected sizing
        img = img.resize((180, 180))
        
        # Convert to array and add batch dimension
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        
        # Check if image file was sent
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400

        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400

        # Read and verify image
        try:
            img = Image.open(io.BytesIO(file.read()))
            logger.info(f"Image opened successfully, format: {img.format}, mode: {img.mode}")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return jsonify({
                'error': 'Invalid image file',
                'success': False
            }), 400

        # Preprocess image
        try:
            processed_image = preprocess_image(img)
            logger.info("Image preprocessed successfully")
        except ValueError as ve:
            logger.error(f"Image validation failed: {str(ve)}")
            return jsonify({
                'error': str(ve),
                'success': False
            }), 400
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return jsonify({
                'error': 'Error processing image',
                'success': False
            }), 500

        # Make prediction
        try:
            prediction = model.predict(processed_image, verbose=0)
            logger.info("Prediction made successfully")
            
            # Log all class probabilities for debugging
            for idx, prob in enumerate(prediction[0]):
                logger.info(f"Class {class_names[idx]}: {prob:.4f}")
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({
                'error': 'Error making prediction',
                'success': False
            }), 500

        # Get class with highest probability
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        class_name = class_names[class_idx]
        
        # Get top 3 predictions for reference
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': class_names[idx],
                'confidence': float(prediction[0][idx])
            }
            for idx in top_3_idx
        ]
        
        logger.info(f"Prediction complete: {class_name} with confidence {confidence}")
        
        return jsonify({
            'prediction': class_name,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5008)  # Changed port to 5006
