import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    try:
        # Load the model
        model = load_model('model.h5')
        logger.info("Model loaded successfully")
        
        # Create a random test image (simulating a normalized image)
        test_image = np.random.rand(1, 180, 180, 3)
        
        # Make prediction
        prediction = model.predict(test_image, verbose=0)
        
        # Print model summary
        model.summary()
        
        # Print prediction distribution
        logger.info("\nPrediction distribution:")
        class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 
                      'melanoma', 'nevus', 'pigmented benign keratosis', 
                      'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
        
        for idx, prob in enumerate(prediction[0]):
            logger.info(f"{class_names[idx]}: {prob:.4f}")
            
        # Print model configuration
        logger.info("\nModel configuration:")
        for layer in model.layers:
            logger.info(f"Layer: {layer.name}, Output shape: {layer.output_shape}")
            
        # Check if the final layer has proper activation
        final_layer = model.layers[-1]
        logger.info(f"\nFinal layer activation: {final_layer.activation.__name__}")
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()
