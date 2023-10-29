import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define constants
BATCH_SIZE = 32

# Set the path to the trained model
model_path = 'skin_condition_model.h5'  # Provide the correct path to your trained model

# Set the path to the test data
test_data_dir = 'dataset\\test_data'  # Provide the correct path to your test data

# Load the trained model
model = load_model(model_path)

# Create a data generator for test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # No data augmentation for testing

# Generate batches of test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),  # Specify the target image size
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Do not shuffle test data
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
