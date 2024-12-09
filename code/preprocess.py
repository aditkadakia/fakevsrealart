import os
import random
from matplotlib import pyplot as plt
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.metrics import Precision, Recall

import shutil

# Paths
DATASET_PATH = "fakevsrealart/Real_AI_SD_LD_Dataset"  # Replace with the actual path
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMAGE_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32

def preprocess_data():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2  # Reserve 20% for validation
    )

    train_generator = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # Multi-class classification
        subset="training",
    )

    validation_generator = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # Multi-class classification
        subset="validation",
    )

    test_generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        TEST_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # Multi-class classification
    )

    print("Class Indices:", train_generator.class_indices)

    return train_generator, validation_generator, test_generator









