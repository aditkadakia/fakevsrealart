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
DATASET_PATH = "path_to_dataset"  # Replace with actual dataset path
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
PROCESSED_TRAIN_PATH = "processed_train"
PROCESSED_TEST_PATH = "processed_test"

IMAGE_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32

def map_classes(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        os.makedirs(os.path.join(dest_dir, "AI"))
        os.makedirs(os.path.join(dest_dir, "Human"))

    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)

        if os.path.isdir(class_path):
            # Determine if AI-generated or human-generated
            if class_dir.startswith("AI_"):
                target_dir = os.path.join(dest_dir, "AI")
            else:
                target_dir = os.path.join(dest_dir, "Human")

            # Copy files to the respective directory
            for file in os.listdir(class_path):
                src_file = os.path.join(class_path, file)
                dest_file = os.path.join(target_dir, file)
                shutil.copy(src_file, dest_file)

def preprocess_data():
    # Map classes to binary folders for train and test
    print("Mapping train classes...")
    map_classes(TRAIN_PATH, PROCESSED_TRAIN_PATH)
    print("Mapping test classes...")
    map_classes(TEST_PATH, PROCESSED_TEST_PATH)

    # Initialize data generators with standardization
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        PROCESSED_TRAIN_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    test_generator = datagen.flow_from_directory(
        PROCESSED_TEST_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    return train_generator, test_generator

if __name__ == "__main__":
    preprocess_data()








