import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import ResNet50
from preprocess import preprocess_data
from preprocess import restructure_dataset
from keras.preprocessing.image import ImageDataGenerator

#make own hp class

class YourModel(tf.keras.Model):
    """ Your own neural network model for binary classification. """

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.architecture = [
            Conv2D(filters=8, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 1)),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.2),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.2),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(units=512, activation="relu"),
            Dense(units=1, activation="sigmoid"),  # Binary classification
        ]

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.BinaryCrossentropy()(labels, predictions)



class ResNetModel(tf.keras.Model):
    def create_resnet_model(self, input_shape):
        # Load ResNet50 pre-trained on ImageNet
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

        # Freeze the base model layers
        base_model.trainable = False

        # Add custom layers for binary classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation="sigmoid")(x)  # Binary classification (AI vs. Human)

        # Create the model
        model = Model(inputs=base_model.input, outputs=output)
        return model

    def train_model(self):
        # Define paths and preprocessing parameters
        TRAIN_PATH = "restructured_train"
        TEST_PATH = "restructured_test"
        IMAGE_SIZE = (128, 128)
        BATCH_SIZE = 32

        # Initialize ImageDataGenerator
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2  # Reserve 20% of training data for validation
        )

        # Create training and validation generators
        train_generator = datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",  # Binary classification
            subset="training",
        )

        validation_generator = datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",  # Binary classification
            subset="validation",
        )

        # Create test generator
        test_generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
            TEST_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",  # Binary classification
        )

        # Print class indices for debugging
        print("Class Indices:", train_generator.class_indices)

        # Create the ResNet model
        input_shape = (128, 128, 3)
        model = self.create_resnet_model(input_shape)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),  # Binary cross-entropy
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10,
            verbose=1,
        )

        # Save the trained model
        model.save("art_classifier_resnet_binary.h5")
        print("Model saved as art_classifier_resnet_binary.h5")



