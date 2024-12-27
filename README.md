# Face Detection & Recognition on Pins FR Dataset

This project performs face detection and recognition using a custom Convolutional Neural Network (CNN) architecture with Transfer Learning, leveraging the Pins Face Recognition (FR) dataset. It identifies and crops faces from images and then trains a model for face recognition using the VGGFace model.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Steps](#steps)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Face Detection & Cropping](#2-face-detection--cropping)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Training the Model](#4-training-the-model)
  - [5. Inference & Evaluation](#5-inference--evaluation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview
The objective of this project is to detect and recognize faces in the Pins FR Dataset using CNNs and Transfer Learning with VGGFace for feature extraction. The project workflow includes data preprocessing, face detection, feature extraction, model training, and evaluation.

## Dataset
The dataset used in this project is the [Pins Face Recognition Dataset](https://www.kaggle.com/datasets/). It includes labeled images of various public figures, making it ideal for face recognition tasks.

## Requirements
Install the necessary libraries with the following commands:
```bash
pip install kaggle opencv-python torch torchvision torchaudio pandas numpy tqdm matplotlib tensorflow
```
## Steps
1. Data Preprocessing
Load images from the dataset.
Resize images to a fixed size of 128x128 pixels.
Normalize pixel values to the range [0, 1].
2. Face Detection & Cropping
Use OpenCV's Haar Cascade for face detection.
Crop faces from the dataset to create a uniform input for the recognition model.
3. Feature Extraction
Utilize the VGGFace model to extract feature vectors for each detected face.
Generate 2622-dimensional embeddings representing each face.
4. Training the Model
Train a classifier on the extracted feature embeddings using CNN layers.
Split the data into training, validation, and test sets for better evaluation.
5. Inference & Evaluation
Test the model with unseen images.
Evaluate accuracy, recall, precision, and F1-score.
