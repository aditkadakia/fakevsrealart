import shutil
import os

IMAGE_SIZE = (128, 128)  
BATCH_SIZE = 32

DATASET_PATH = "Real_AI_SD_LD_Dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

def restructure_dataset(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, "AI"))
        os.makedirs(os.path.join(target_dir, "Human"))

    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)

        if not os.path.isdir(class_path):
            continue

        #dividing the dataset based on whether it is AI or Human 
        if "AI_" in class_dir:
            dest_dir = os.path.join(target_dir, "AI")
        else:
            dest_dir = os.path.join(target_dir, "Human")
            
        for file in os.listdir(class_path):
            src_file = os.path.join(class_path, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy(src_file, dest_file)

# Restructure train and test sets
restructure_dataset(TRAIN_PATH, "restructured_train")
restructure_dataset(TEST_PATH, "restructured_test")








