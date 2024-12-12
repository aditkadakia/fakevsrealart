from model import ResNetModel
import os
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import shutil



def main():
    resnet = ResNetModel()
    model_path = "art_classifier_resnet_binary.h5"

    # Train or load the model
    if not os.path.exists(model_path):
        print("Training model...")
        resnet.train_model()
    else:
        print(f"Loading pre-trained model from {model_path}...")

    # # Load the trained model
    # model = models.load_model(model_path)
    # save_misclassified_images(model, "restructured_test", "misclassified_images")

    # # LIME explanations
    # image_path = "path/to/your/image.jpg"
    # if not os.path.exists(image_path):
    #     raise FileNotFoundError(f"Image not found at {image_path}")

    # def preprocess_fn(image):
    #     return image / 255.0  # Normalize to [0, 1]

    # timestamp = "lime_run_001"
    # print("Generating LIME explanations...")
    # LIME_explainer(model, image_path, preprocess_fn, timestamp)

# def save_misclassified_images(model, test_path, output_dir):
#     """
#     Evaluate the model on the test set and save misclassified images.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Prepare the test data generator
#     test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
#     test_generator = test_datagen.flow_from_directory(
#         test_path,
#         target_size=(128, 128),
#         batch_size=1,  # Evaluate one image at a time
#         class_mode="binary",
#         shuffle=False,
#     )

#     # Iterate through the test dataset
#     for i in range(len(test_generator)):
#         image, label = test_generator[i]
#         prediction = model.predict(image)
#         predicted_label = 1 if prediction > 0.5 else 0

#         # If misclassified, save the image
#         if predicted_label != int(label[0]):
#             # Get the original file path
#             original_image_path = test_generator.filepaths[i]
#             filename = os.path.basename(original_image_path)

#             # Save the misclassified image
#             dest_path = os.path.join(output_dir, filename)
#             shutil.copy(original_image_path, dest_path)
#             print(f"Saved misclassified image: {filename}")

# def LIME_explainer(model, path, preprocess_fn, timestamp):
#     """
#     This function takes in a trained model and a path to an image and outputs 4
#     visual explanations using the LIME model.
#     """

#     # Create directories for saving results
#     save_directory = f"lime_explainer_images{os.sep}{timestamp}"
#     os.makedirs(save_directory, exist_ok=True)

#     # Helper function to save and display explanations
#     def image_and_mask(explanation, title, positive_only=True, num_features=5, hide_rest=True, idx=0):
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0], positive_only=positive_only, num_features=num_features, hide_rest=hide_rest
#         )
#         plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         plt.title(title)
#         image_save_path = os.path.join(save_directory, f"{idx}.png")
#         plt.savefig(image_save_path, dpi=300, bbox_inches="tight")
#         plt.show()

#     # Load and preprocess the image
#     image = imread(path)
#     if len(image.shape) == 2:  # Grayscale image
#         image = np.stack([image] * 3, axis=-1)  # Convert to 3 channels
#     image_resized = resize(image, (128, 128, 3), preserve_range=True)
#     image_preprocessed = preprocess_fn(image_resized)

#     # Ensure the input matches the model's expected shape
#     image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

#     # Initialize LIME explainer
#     explainer = lime_image.LimeImageExplainer()

#     # Generate explanation
#     explanation = explainer.explain_instance(
#         image_resized.astype("double"),
#         model.predict,
#         top_labels=5,
#         hide_color=0,
#         num_samples=1000,
#     )

#     # Save visualizations
#     image_and_mask(explanation, "Top 5 superpixels", positive_only=True, num_features=5, hide_rest=True, idx=0)
#     image_and_mask(explanation, "Top 5 with the rest of the image", positive_only=True, num_features=5, hide_rest=False, idx=1)
#     image_and_mask(explanation, "Pros (green) and Cons (red)", positive_only=False, num_features=10, hide_rest=False, idx=2)

#     # Heatmap of superpixel weights
#     ind = explanation.top_labels[0]
#     dict_heatmap = dict(explanation.local_exp[ind])
#     heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
#     plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
#     plt.colorbar()
#     plt.title("Heatmap of superpixel weights")
#     heatmap_save_path = os.path.join(save_directory, "heatmap.png")
#     plt.savefig(heatmap_save_path, dpi=300, bbox_inches="tight")
#     plt.show()

#     print(f"LIME explanations saved in {save_directory}")


if __name__ == "__main__":
    main()