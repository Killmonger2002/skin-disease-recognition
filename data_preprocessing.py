import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to the train, validation, and test data
train_data_dir = 'dataset\\train_data'
validation_data_dir = 'dataset\\validation_data'
test_data_dir = 'dataset\\test_data'

# Set parameters for data augmentation
batch_size = 32
image_size = (224, 224)  # The size of input images

# Create data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to the range [0, 1]
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift the width of images
    height_shift_range=0.2,  # Randomly shift the height of images
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Randomly zoom in on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill empty areas when applying transformations
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)  # No data augmentation for validation

# Create data generators for training, validation, and testing
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
