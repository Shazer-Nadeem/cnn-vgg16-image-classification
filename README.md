# cnn-vgg16-image-classification

This repository contains a Jupyter Notebook implementation of an **image classification pipeline** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning with VGG16**. The project demonstrates dataset preprocessing, model training, evaluation, and visualization of results.

---

## 📂 Project Structure
- `i222043_A.ipynb` — Main notebook containing the full workflow  
- `Dataset.zip` — Input dataset (images organized by class folders)  
- `Dataset/` — Extracted dataset directory  

---

## 🚀 Features
- Dataset extraction and preprocessing with **NumPy** and **PIL**  
- Train/test splitting with **Scikit-learn**  
- Custom CNN model built with **Keras Sequential API**  
- Transfer Learning using **VGG16**  
- Model evaluation with **classification reports, confusion matrix, and accuracy/loss plots**  
- Regularization with **Dropout**, **Batch Normalization**, and **Early Stopping**  

---

## 📦 Requirements
Install dependencies before running the notebook:
pip install numpy matplotlib pillow scikit-learn seaborn tensorflow
▶️ Usage
Clone this repository:

## Git
git clone https://github.com/yourusername/cnn-vgg16-image-classification.git
cd cnn-vgg16-image-classification
Place your dataset as a zip file named Dataset.zip in the working directory.

##Open and run the notebook:

jupyter notebook i222043_A.ipynb
The notebook will:

##Extract the dataset

Train the CNN and VGG16 models

Display training curves and evaluation metrics

##📊 Results
Accuracy and loss plots over epochs

Confusion matrix heatmap

Classification report (precision, recall, F1-score)

##📌 Notes
The dataset should be structured as:

mathematica
Copy code
Dataset/
  ├── Class1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ├── Class2/
  │   ├── img3.jpg
  │   ├── img4.jpg
You can adjust hyperparameters (learning rate, batch size, epochs) inside the notebook

##🛠️ Tech Stack
Python 3.x

TensorFlow / Keras

Scikit-learn

NumPy, Matplotlib, Seaborn
