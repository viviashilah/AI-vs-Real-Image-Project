import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================================================
# LOAD MODEL
# ==================================================
model = load_model(r'D:\Porto-project\AI vs Real-Image\model\mobilenetv2_model.h5')

# ==================================================
# IMAGE SETTINGS
# ==================================================
IMG_SIZE = 224
BATCH_SIZE = 32

# ==================================================
# LOAD TEST DATASET
# ==================================================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    r'D:\Porto-project\AI vs Real-Image\dataset\final_dataset\test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==================================================
# EVALUATE MODEL
# ==================================================
loss, accuracy = model.evaluate(test_generator)

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# ==================================================
# PREDICTION
# ==================================================
predictions = model.predict(test_generator)

predicted_classes = (predictions > 0.5).astype(int)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

# ==================================================
# CLASSIFICATION REPORT
# ==================================================
print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels
    )
)

# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(true_classes, predicted_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(ax=ax)

plt.title('Confusion Matrix')

plt.savefig('assets/confusion_matrix.png')

plt.show()

print("\nConfusion matrix saved to assets/confusion_matrix.png")

# ==================================================
# SAMPLE PREDICTIONS
# ==================================================
print("\n===================================")
print("SAMPLE PREDICTIONS")
print("===================================")

for i in range(10):

    prediction = predicted_classes[i][0]

    actual = true_classes[i]

    pred_label = class_labels[prediction]

    actual_label = class_labels[actual]

    print(f"Image {i+1}")
    print(f"Actual    : {actual_label}")
    print(f"Predicted : {pred_label}")
    print("-----------------------------------")

print("\nEvaluation completed successfully!")