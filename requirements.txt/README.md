tensorflow==2.13.0
numpy==1.24.3
pillow
scikit-learn==1.3.2
matplotlib==3.7.5

# Satellite Image Classification using Deep Learning

## Overview
This project classifies satellite images into four categories:
- Forest
- Highway
- Industrial
- River

The model is built using Teachable Machine and evaluated using Python.

---

## Dataset
EuroSAT dataset (2750 images per class subset used for testing)

---

## Model
- Framework: TensorFlow
- Input size: 224x224 RGB images
- Model type: Teachable Machine exported CNN

---

## Results

- Accuracy: 85.47%

### Key Observations
- Strong performance on Forest and Industrial classes
- Misclassification mainly between Highway and River

---

## Confusion Matrix

*(Insert image from results folder here)*

---

## Project Structure
- `model/` → trained model files
- `scripts/` → prediction and evaluation code
- `results/` → output graphs and reports

---

## How to Run

### 1. Install dependencies

### 2. Run evaluation

### 3. Run single prediction
python scripts/predict_single.py


---

## Future Improvements
- Replace Teachable Machine model with ResNet50
- Improve Highway vs River classification
- Train on full EuroSAT dataset

plt.savefig("results/confusion_matrix.png")

---

## Author
Electrical Engineering Graduate with experience in AI, Energy Automation in Buildings, and machine learning projects.
