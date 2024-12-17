import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import os


class ResNetModel(tf.keras.Model):
    def create_resnet_model(self, input_shape):
        # laoading ResNet50 which is pre-trained on ImageNet
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False #freeze the base layers 

        # custom layers for binary classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation="sigmoid")(x) 

        model = Model(inputs=base_model.input, outputs=output)
        return model

    def train_model(self):
        # definitions for paths and parameters 
        TRAIN_PATH = "restructured_train"
        IMAGE_SIZE = (128, 128)
        BATCH_SIZE = 32
        CHECKPOINT_DIR = "checkpoints/resnet_model"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # initalizing data 
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2  # 20% training data used for validation 
        )

        # training generator 
        train_generator = datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",  
            subset="training",
        )

        #validation generator 
        validation_generator = datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",  # Binary classification
            subset="validation",
        )

        input_shape = (128, 128, 3)
        model = self.create_resnet_model(input_shape)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),  # Binary cross-entropy
            metrics=["accuracy"],
        )

        #saving checkpoints for possible reruning from best checkpoints
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "resnet_model_epoch_{epoch:02d}.h5"),
            save_weights_only=False,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        )

        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=50,
            callbacks=[checkpoint_callback],
            verbose=1,
        )

        model.save("art_classifier_resnet_binary.h5")
        print("Model saved as art_classifier_resnet_binary.h5")



