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
import os

# Paths
<<<<<<< HEAD
DATASET_PATH = "Real_AI_SD_LD_Dataset"  # Replace with the actual path
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
=======
# DATASET_PATH = "fakevsrealart/Real_AI_SD_LD_Dataset"  # Replace with the actual path
# TRAIN_PATH = os.path.join(DATASET_PATH, "train")
# TEST_PATH = os.path.join(DATASET_PATH, "test")
>>>>>>> bb94bab5151a6ec0fac817f62197ce8c7f01dc51

IMAGE_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32

# def preprocess_data():
#     datagen = ImageDataGenerator(
#         rescale=1.0 / 255.0,
#         validation_split=0.2  # Reserve 20% for validation
#     )

#     train_generator = datagen.flow_from_directory(
#         TRAIN_PATH,
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="categorical",  # Multi-class classification
#         subset="training",
#     )

#     validation_generator = datagen.flow_from_directory(
#         TRAIN_PATH,
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="categorical",  # Multi-class classification
#         subset="validation",
#     )

#     test_generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
#         TEST_PATH,
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="categorical",  # Multi-class classification
#     )

#     print("Class Indices:", train_generator.class_indices)

#     return train_generator, validation_generator, test_generator

DATASET_PATH = "fakevsrealart/Real_AI_SD_LD_Dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

def restructure_dataset(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, "AI"))
        os.makedirs(os.path.join(target_dir, "Human"))

    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)

        # Check if it's an AI class or Human class
        if "AI_" in class_dir:
            dest_dir = os.path.join(target_dir, "AI")
        else:
            dest_dir = os.path.join(target_dir, "Human")

        # Move images to the target directory
        for file in os.listdir(class_path):
            src_file = os.path.join(class_path, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy(src_file, dest_file)

# Restructure train and test sets
restructure_dataset(TRAIN_PATH, "restructured_train")
restructure_dataset(TEST_PATH, "restructured_test")








