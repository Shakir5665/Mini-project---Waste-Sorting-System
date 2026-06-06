# ♻️ Waste-Sorting Image Classification System
An Intelligent System Mini-Project by team **TECH DREAMERS** (ICT 3212 - Intelligent Systems).

This repository contains a two-stage deep learning implementation for classifying household waste into **Organic** and **Recyclable** categories using Convolutional Neural Networks (CNNs). The project aims to provide a reliable, automated computer vision component for smart waste-bin sorting systems.

---

## 📌 Table of Contents
1. [Project Evolution](#-project-evolution)
2. [Comparative Analysis (Implementation 1 vs. 2)](#-comparative-analysis-implementation-1-vs-2)
3. [Dataset Architecture](#-dataset-architecture)
4. [Model Architectures](#-model-architectures)
5. [Training & Regularization Strategy](#-training--regularization-strategy)
6. [Results & Metrics](#-results--metrics)
7. [Real-World Inference Pipeline](#-real-world-inference-pipeline)
8. [Installation & Local Setup](#-installation--local-setup)
9. [Team Members](#-team-members)

---

## 🚀 Project Evolution
The project was designed iteratively to address common failure modes in machine learning pipelines, such as data leakage, overfitting, and poor generalization:
*   **Implementation 1 (Baseline):** Established the feasibility of binary waste classification using a simple, shallow CNN model on a small dataset with a naive validation split strategy.
*   **Implementation 2 (Optimized System):** Solved generalization and overfitting problems by restructuring the data directory, implementing live image augmentation, adding dropout layers, deepening the model structure, and integrating early stopping mechanisms. It also adds an interactive utility to test the trained model on zip archives of real-world images.

---

## 📊 Comparative Analysis (Implementation 1 vs. 2)

| Feature | Implementation 1 (Baseline) | Implementation 2 (Optimized) | Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Data Split Strategy** | Dynamic 15% validation split from a single folder. | Explicit, physical train, validation, and test split. | Prevents validation data leakage and ensures stable testing. |
| **Total Images** | 1,335 images (1,135 Train / 200 Val) | 7,042 images (4,462 Train / 900 Val / 1,680 Test) | Larger dataset size dramatically improves model generalization. |
| **Data Augmentation** | None (only pixel rescaling `1./255`) | Rotation ($20^\circ$), zoom ($0.2$), shear ($0.2$), horizontal flip. | Regularizes the model to be invariant to rotation, zoom, and orientation. |
| **Convolutional Layers** | 3 layers with filters: `16 -> 32 -> 64` | 3 layers with filters: `32 -> 64 -> 128` | Higher capacity to capture rich visual features and shapes. |
| **Fully Connected Layers** | 1 Dense layer (`64` units) | 1 Dense layer (`128` units) | Learns more complex non-linear combinations of features. |
| **Regularization** | None | Dropout layer (`rate = 0.5`) | Disables random neurons to prevent co-adaptation and overfitting. |
| **Training Epochs** | Fixed `8` Epochs | Up to `20` Epochs with Early Stopping | Halts training when validation loss stops improving (patience = 3). |
| **Evaluation Data** | Validation set | Dedicated, unseen Test set | Gives a true performance metric under realistic conditions. |

---

## 🗂️ Dataset Architecture
The raw waste images are organized as follows in Implementation 2:
```
📂 dataset_Implementation-2/
├── 📂 Train/         # 4,462 images (used to train parameters)
│   ├── 📂 Organic/
│   └── 📂 Recyclable/
├── 📂 Validation/    # 900 images (used for tuning hyper-parameters and early stopping)
│   ├── 📂 Organic/
│   └── 📂 Recyclable/
└── 📂 Test/          # 1,680 images (completely unseen, for final evaluation)
    ├── 📂 Organic/
    └── 📂 Recyclable/
```
*   **Classes:** `Organic` (labeled 0) and `Recyclable` (labeled 1).
*   **Image Dimensions:** Scaled to $128 \times 128$ pixels with 3 channels (RGB).

---

## 🏗️ Model Architectures

### Implementation 1 Model
A lightweight network designed for speed and initial feasibility checks:
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Implementation 2 Model
A high-capacity model designed for production deployment and high-accuracy requirements:
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(1, activation='sigmoid')
])
```

---

## ⚙️ Training & Regularization Strategy
1.  **Image Augmentation Configuration (Implementation 2):**
    ```python
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )
    ```
2.  **Early Stopping Hook:**
    Monitors validation loss (`val_loss`) with a patience of `3`. It restores the best weight set automatically if the validation loss degrades for three consecutive epochs:
    ```python
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    ```

---

## 📊 Results & Metrics

### Model Performance Summary
*   **Implementation 1 Validation Accuracy:** ~92% (on a subset of training data).
*   **Implementation 2 Test Accuracy:** **95.30%** (evaluated on the completely isolated Test split).

### Confusion Matrix (Implementation 2)
The classification report and confusion matrix below show strong performance, balance, and clean class separation:
```
[[ True Organic   False Recyclable ]
 [ False Organic  True Recyclable  ]]
```
*(You can find the interactive Confusion Matrix plot rendered directly inside the Jupyter Notebook).*

---

## 🔮 Real-World Inference Pipeline
Implementation 2 introduces a dedicated inference segment for classifying real-world images from zip folders:
1.  **Model Loading:** Reloads the best-saved weights (`waste_classification_model_v2.h5`).
2.  **Zip Upload:** Uploads a `.zip` archive containing any waste images (`.jpg`, `.jpeg`, `.png`).
3.  **Extraction & Prediction:** Unpacks the archive, processes the images, and predicts classes:
    *   **$\text{Probability} \ge 0.5$:** `RECYCLABLE` (displayed in green)
    *   **$\text{Probability} < 0.5$:** `ORGANIC` (displayed in red)
4.  **Grid Visualization:** Displays the input images along with classification labels and prediction confidence.
5.  **Summary Table:** Prints a tabular summary of image filenames, predictions, and counts overall statistics.

---

## 💻 Installation & Local Setup

### Prerequisites
*   Python 3.8 or above
*   pip package manager
*   GPU acceleration (CUDA & cuDNN) is highly recommended for faster training

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Shakir5665/Mini-project---Waste-Sorting-System.git
    cd Mini-project---Waste-Sorting-System
    ```
2.  **Install Dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib seaborn scikit-learn
    ```
3.  **Run the Notebooks:**
    You can run the models in your local environment or upload them to Google Colab:
    *   `Mini_project_Implementation_1.ipynb`
    *   `Mini_project_Implementation_2.ipynb`

---

## 👥 Team Members

**TECH DREAMERS**

| Name | Registration Number | Index Number |
| :--- | :--- | :--- |
| **MNM.SAKIR** | ICT/2022/059 | 5665 |
| **MI.AFTHAL AHAMED** | ICT/2022/105 | 5708 |
| **AW.IMADH AHMED** | ICT/2022/122 | 5724 |
| **NM.BAAHIR** | ICT/2022/045 | 5651 |
| **MM.RAASHIDH** | ICT/2022/135 | 5736 |

---
*Developed as part of the Intelligent Systems Coursework.*
