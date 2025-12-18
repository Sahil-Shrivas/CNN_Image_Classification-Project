# 🐱🐶 Cat vs Dog Image Classification using CNN  
A Deep Learning Project for Binary Image Classification

---

## 🌟 **Project Description**

This project implements a **Convolutional Neural Network (CNN)** to classify images of **cats** and **dogs**.  
The goal is to build an **end-to-end deep learning pipeline** that covers:

- Data preprocessing  
- CNN architecture building  
- Model training  
- Validation & testing  
- Prediction on new/unseen images  

The repository is structured to help beginners and intermediate learners understand **how CNNs work**, how image data is prepared, and how a trained model can be used for real-world image classification.
This problem is popular in machine learning because it teaches all core concepts of deep learning applied to computer vision — from convolution operations to feature extraction and binary classification.

---

## **Screenshot**

![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/CNN_Image_Classification-Project/refs/heads/main/Screenshot%202025-12-18%20203620.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/CNN_Image_Classification-Project/refs/heads/main/Screenshot%202025-12-18%20203634.png)
![alt text](https://raw.githubusercontent.com/Sahil-Shrivas/CNN_Image_Classification-Project/refs/heads/main/Screenshot%202025-12-18%20203651.png)

---

# 📘 **Key Concepts Explained**

Below are important keywords used throughout this project:

### 🔹 **1. CNN (Convolutional Neural Network)**  
A CNN is a type of deep learning model designed to work with **images**.  
It uses convolution operations to automatically extract important features like edges, shapes, textures.

**Why CNN for image classification?**  
Because it learns features directly from raw images without manual feature engineering.

---

### 🔹 **2. Binary Classification**  
This project has only **two classes**:  
- Cat  
- Dog  

Binary classification means predicting one of two possible outcomes.

---

### 🔹 **3. Convolution Layer**  
A layer that applies filters (kernels) to the input image.  
This helps detect local patterns such as:

- Edges  
- Corners  
- Color transitions  
- Shapes  

---

### 🔹 **4. Pooling Layer**  
Reduces the size of the feature map.  
This helps to:

- Decrease computation  
- Reduce overfitting  
- Keep important features  

---

### 🔹 **5. Flattening**  
Converts a 2D feature map into a 1D vector so it can be fed into fully connected layers (Dense layers).

---

### 🔹 **6. Dense Layer (Fully Connected Layer)**  
A layer where each neuron connects to all neurons in the previous layer.  
Used for final classification.

---

### 🔹 **7. Activation Function**  
Adds non-linearity to the network.  
Common ones used here:

- **ReLU** — for hidden layers  
- **Sigmoid** — for binary classification output  

---

### 🔹 **8. Data Preprocessing**  
Before training, all images must be:

- Resized  
- Normalized  
- Augmented (optional)  

This ensures stable and accurate training.

---

### 🔹 **9. Model Evaluation**  
After training, the model is tested using:

- Accuracy  
- Loss  
- Confusion matrix  
- Predictions  

---

### 🔹 **10. Inference**  
Using the trained model to classify new images (cat/dog).

---

## 🐾 Make Predictions
Use the trained model to classify new images as:
- `0` → Cat 😺  
- `1` → Dog 🐶  

---

# 📂 **Repository Structure**
    CatVSDog-Image-Classification-Project/
    │
    ├── data/ # Dataset (if included)
    ├── notebooks/ # Jupyter notebooks for training & testing
    │ └── cat_dog_classifier.ipynb
    │
    ├── src/ # Python modules for modular codebase
    │ ├── data_loader.py # Loads and preprocesses images
    │ ├── model.py # CNN architecture
    │ └── train.py # Training loop
    │
    ├── saved_models/ # Trained model files (.h5 or .pth)
    ├── requirements.txt # Required Python libraries
    └── README.md

---

# ⚙️ **Installation & Setup**

### **Clone the Repository**
```bash
git clone https://github.com/Sahil-Shrivas/CatVSDog-Image-Classification-Project.git
cd CatVSDog-Image-Classification-Project
```
---

## 📬 Contact
👤 **Sahil Shrivas**  
🔗 GitHub: https://github.com/Sahil-Shrivas

