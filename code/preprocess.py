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
DATASET_PATH = "path_to_dataset"  # Replace with the actual path
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMAGE_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32

def preprocess_data():
    # Initialize data generators
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",  # Binary classification
    )

    test_generator = datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    return train_generator, test_generator

if __name__ == "__main__":
    preprocess_data()









