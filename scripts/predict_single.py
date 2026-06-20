 import tensorflow as tf
import numpy as np
from PIL import Image

# ── Step 1: Load the Model ──────────────────────────────────────────────────
model = tf.keras.models.load_model(
    r"D:\Documents\Workshop\Teachable Machine\Model testing result\converted_keras\keras_model.h5"
)
print("Model loaded successfully!")

# ── Step 2: Define Class Names ──────────────────────────────────────────────
class_names = ["Forest", "Highway", "Industrial", "River"]

# ── Step 3: Image Preparation Function ─────────────────────────────────────
def prepare_image(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array

# ── Step 4: Load 4 Images (one from each class) ─────────────────────────────
img1 = prepare_image(r"D:\Documents\Workshop\Teachable Machine\EuroSAT\2750\Forest\Forest_1675.jpg")
img2 = prepare_image(r"D:\Documents\Workshop\Teachable Machine\EuroSAT\2750\Highway\Highway_2405.jpg")
img3 = prepare_image(r"D:\Documents\Workshop\Teachable Machine\EuroSAT\2750\Industrial\Industrial_1777.jpg")
img4 = prepare_image(r"D:\Documents\Workshop\Teachable Machine\EuroSAT\2750\River\River_1875.jpg")

# ── Step 5: Stack into One Batch ────────────────────────────────────────────
batch = np.stack([img1, img2, img3, img4], axis=0)

print("Batch shape:", batch.shape)
# Expected output → (4, 224, 224, 3)

# ── Step 6: Run Prediction on Entire Batch ──────────────────────────────────
predictions = model.predict(batch)

print("\nRaw Predictions:")
print(predictions)

# ── Step 7: Display Results for Each Image ──────────────────────────────────
print("\n" + "="*50)
print("        PREDICTION RESULTS — BATCH OF 4")
print("="*50)

image_paths = [
    r"Forest\Forest_1675.jpg",
    r"Highway\Highway_2405.jpg",
    r"Industrial\Industrial_1777.jpg",
    r"River\River_1875.jpg"
]

for i, pred in enumerate(predictions):
    predicted_index = np.argmax(pred)
    confidence = np.max(pred) * 100

    print(f"\nImage {i+1}: {image_paths[i]}")
    print(f"  Predicted Class : {class_names[predicted_index]}")
    print(f"  Confidence      : {confidence:.2f}%")
    print(f"  All Scores      :")
    for j, score in enumerate(pred):
        bar = "█" * int(score * 30)   # visual confidence bar
        print(f"    {class_names[j]:<12} {score*100:6.2f}%  {bar}")

print("\n" + "="*50)

import tensorflow as tf
print(tf.__version__)
