from model import ResNetModel
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import random


def main(run_evaluation = True):
    resnet = ResNetModel()
    model_path = "art_classifier_resnet_binary.h5"
    test_data_dir = "restructured_test"
    image_size = (128, 128)
    batch_size = 32

    if not os.path.exists(model_path):
        print("Training model...")
        resnet.train_model()
    else:
        print(f"Loading pre-trained model from {model_path}...")
        model = load_model(model_path, compile=False)
        model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss=tf.keras.losses.BinaryCrossentropy(),          
            metrics=["accuracy"]                 
        )
    
    test_generator = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",  
        shuffle=False  
    )

    if run_evaluation: 
        print("Evaluating model accuracy...")
        loss, accuracy = model.evaluate(test_generator, verbose=1)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    precision, recall = calculate_precision_recall(model, test_generator)
    display_precision_recall_table(precision, recall)

    misclassified = []
    correctly_classified = []

    random_numbers = [random.randint(0, len(test_generator)) for _ in range(10)]


    for _, i in enumerate(random_numbers):
        image, label = test_generator[i]
        prediction = model.predict(image)
        predicted_label = "AI" if prediction[0][0] > 0.5 else "Human"
        actual_label = "AI" if int(label[0]) == 1 else "Human"

        if predicted_label != actual_label:
            misclassified.append((image, actual_label, predicted_label))
        else:
            correctly_classified.append((image, actual_label, predicted_label))

    selected_images = misclassified + correctly_classified

    for idx, (image, actual_label, predicted_label) in enumerate(selected_images):
        correctly_classified_flag = "Correctly Classified" if predicted_label == actual_label else "Misclassified"
        perform_lime(model, image[0], idx, correctly_classified_flag, predicted_label, actual_label)
    
def perform_lime(model, image, idx, classification_status, predicted_label, actual_label):
    """Perform LIME analysis on a single image."""
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype("double"),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    save_directory = f"lime_explainer_images{os.sep}"
    os.makedirs(save_directory, exist_ok=True)

    def image_and_mask(explanation, title, positive_only=True, num_features=5, hide_rest=True, idx=0):
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only, num_features=num_features, hide_rest=hide_rest
        )
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)
        path = os.path.join(save_directory, f"{idx}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

        # Top features visualization
    title = f"Pros (green) and Cons (red) - {classification_status}: Predicted {predicted_label}, Actual {actual_label}"
    lime_image_path = image_and_mask(explanation, title=title,
                                     positive_only=False, num_features=10, hide_rest=False, idx=idx)


    print(f"LIME explanation saved: {lime_image_path}")

def generate_accuracy_chart(model, test_generator):
    """
    Generate an accuracy chart for the final model based on the test dataset.
    """
    batch_accuracies = []
    running_correct = 0
    running_total = 0

    for batch_idx in range(len(test_generator)):
        images, labels = test_generator[batch_idx]
        predictions = model.predict(images)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        actual_labels = labels.astype(int).flatten()

        batch_correct = (predicted_labels == actual_labels).sum()
        batch_total = len(labels)
        batch_accuracy = batch_correct / batch_total

        running_correct += batch_correct
        running_total += batch_total

        batch_accuracies.append(batch_accuracy)

    overall_accuracy = running_correct / running_total
    print(f"Overall Test Accuracy: {overall_accuracy * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(batch_accuracies) + 1), batch_accuracies, label="Batch Accuracy")
    plt.axhline(y=overall_accuracy, color="red", linestyle="--", label="Overall Accuracy")
    plt.title("Batch-wise Accuracy for the Final Model")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_chart.png")
    plt.show()

def calculate_precision_recall(model, test_generator):
    """
    Calculate precision and recall for the model on the test dataset.
    """
    tp, fp, tn, fn = 0, 0, 0, 0

    for batch_idx in range(len(test_generator)):
        images, labels = test_generator[batch_idx]
        predictions = model.predict(images)
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        actual_labels = labels.astype(int).flatten()

        for predicted, actual in zip(predicted_labels, actual_labels):
            if predicted == 1 and actual == 1:
                tp += 1  
            elif predicted == 1 and actual == 0:
                fp += 1  
            elif predicted == 0 and actual == 0:
                tn += 1 
            elif predicted == 0 and actual == 1:
                fn += 1 

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Confusion Matrix:")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return precision, recall

def display_precision_recall_table(precision, recall):
    """
    Display a simple precision-recall table.
    """
    print("Precision and Recall Table")
    print(f"{'Metric':<12} {'Value':<8}")
    print(f"{'-'*20}")
    print(f"{'Precision':<12} {precision:.2f}")
    print(f"{'Recall':<12} {recall:.2f}")



if __name__ == "__main__":
    main(run_evaluation=False)