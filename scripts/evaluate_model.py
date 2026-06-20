import tensorflow as tf
import numpy as np
from PIL import Image
import os

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =====================
# Load model
# =====================
model = tf.keras.models.load_model(
    r"D:\Documents\Workshop\Teachable Machine\Model testing result\converted_keras\keras_model.h5"
)

# =====================
# Dataset path
# =====================
dataset_path = r"D:\Documents\Workshop\Teachable Machine\EuroSAT\2750"

# =====================
# Labels
# =====================
class_names = ["Forest", "Highway", "Industrial", "River"]

y_true = []
y_pred = []

# =====================
# Prediction loop
# =====================
for class_index, class_name in enumerate(class_names):

    folder = os.path.join(dataset_path, class_name)

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            pred_index = np.argmax(prediction)

            y_true.append(class_index)
            y_pred.append(pred_index)

        except Exception as e:
            print("Error:", img_path, e)

# =====================
# Accuracy
# =====================
accuracy = accuracy_score(y_true, y_pred)

print("\n====================")
print("RESULTS")
print("====================")
print("Total images:", len(y_true))
print("Accuracy:", round(accuracy * 100, 2), "%")

# =====================
# Classification report
# =====================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# =====================
# Confusion matrix
# =====================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# =====================
# Plot confusion matrix
# =====================
plt.figure(figsize=(7, 6))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =========================
# PDF setup
# =========================
pdf = SimpleDocTemplate("Satellite_Image_Classification_Report.pdf", pagesize=A4)

styles = getSampleStyleSheet()
content = []

