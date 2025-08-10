import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model(weights_path=None):
    # Use a pre-trained base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(180, 180, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = models.Sequential([
        # Input layer with preprocessing
        layers.Input(shape=(180, 180, 3)),
        layers.Rescaling(1./255),
        
        # Pre-trained base model
        base_model,
        
        # Add custom classification layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(9, activation='softmax')  # 9 classes
    ])
    
    # Compile with better learning rate and optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Load weights if provided
    if weights_path and tf.io.gfile.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except:
            print(f"Could not load weights from {weights_path}")
    
    return model

if __name__ == "__main__":
    # Create model with pre-trained weights
    model = create_model()
    
    # Save the model
    model.save('model.h5')
    print("Model saved as model.h5")
