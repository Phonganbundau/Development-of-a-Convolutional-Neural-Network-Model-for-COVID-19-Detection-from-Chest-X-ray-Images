
# COVID-19 Detection from Chest X-ray Images using Convolutional Neural Networks

## Authors
- Truong Tan Phong - 201260531
- Luu Minh Nhat - 201260471
- Tran Tat Tri - 201260301

## Abstract
This project aims to develop a Convolutional Neural Network (CNN) model to detect COVID-19 from chest X-ray images, providing a faster and more accessible diagnostic alternative. The dataset includes three classes: Normal, Viral Pneumonia, and COVID-19. The model achieved an accuracy of **95.65%**, showcasing its potential as a supplementary diagnostic tool, especially in resource-limited settings where traditional diagnostic methods like RT-PCR may be less accessible.

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Architecture](#model-architecture)
    - [Loss Function](#loss-function)
    - [L2 Regularization](#l2-regularization)
    - [Softmax Function](#softmax-function)
3. [Experiments](#experiments)
    - [Dataset](#dataset)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Experimental Setup](#experimental-setup)
    - [Experimental Results](#experimental-results)
4. [How to Run the Code](#how-to-run-the-code)
5. [Conclusion](#conclusion)

## Introduction
The COVID-19 pandemic created an urgent need for rapid and efficient diagnostic tools. Traditional methods such as RT-PCR are effective but often slow and resource-intensive. Chest X-rays provide a faster alternative, and with the advancements in deep learning, CNNs are used to identify complex patterns in medical images, improving diagnostic accuracy.

This project develops a CNN model capable of classifying chest X-ray images into three categories: **Normal**, **Viral Pneumonia**, and **COVID-19**, offering a powerful supplementary diagnostic tool in clinical settings.

## Methodology

### Data Preprocessing
- **Image Resizing**: All chest X-ray images were resized to 224x224 pixels.
- **Normalization**: Images were normalized to a pixel value range of 0 to 1.
- **Dataset**: The dataset includes chest X-ray images of Normal, Viral Pneumonia, and COVID-19 cases.

### Model Architecture
- The CNN consists of four convolutional layers, each followed by batch normalization and max pooling layers. The number of filters increases progressively from 32 to 256 across these layers.
- The final fully connected layers (512, 256, and 3 neurons) use dropout to prevent overfitting, with the softmax activation function in the output layer for multi-class classification.

### Loss Function
We used the **Cross-Entropy Loss** function, which is well-suited for multi-class classification. It measures the dissimilarity between predicted probabilities and true labels.

### L2 Regularization
L2 regularization was applied to reduce overfitting by penalizing large model coefficients, enhancing the model's generalization ability.

### Softmax Function
The softmax function was applied to the output layer to convert the raw logits into probabilities for each class (Normal, Viral Pneumonia, and COVID-19).

## Experiments

### Dataset
- The dataset includes **6432 chest X-ray images**, categorized into three classes: Normal, Viral Pneumonia, and COVID-19.
- The dataset was split into **80% training** and **20% testing** to evaluate the model’s performance.

### Evaluation Metrics
- **Accuracy**: 95.65%
- **Precision**, **Recall**, and **F1-Score** were calculated for each class (COVID-19, Normal, and Viral Pneumonia).

### Experimental Setup
- **Hardware**: NVIDIA GeForce RTX 3050, Intel Core i7-11370H, 16GB RAM.
- **Software**: Python 3.10.11, Jupyter Notebook.
- **Training Parameters**: 
  - Optimizer: Adam, learning rate 0.001.
  - Epochs: 30
  - Batch size: 32
  - Early stopping and regularization techniques (dropout, L2 regularization) were used.

### Experimental Results
- The model achieved a high accuracy of **95.65%**, with excellent performance in distinguishing between COVID-19 and Viral Pneumonia cases.
- Comparison with ResNet showed that our model outperforms baseline models across key metrics.

## How to Run the Code

### Required Libraries
Install the required libraries using the command below:

```bash
pip install -r requirements.txt
```

Libraries required:
- torch (PyTorch)
- seaborn
- matplotlib
- scikit-learn
- torchvision
- numpy

### Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia?fbclid=IwY2xjawE7G-tleHRuA2FlbQIxMAABHZYkYj0wA9q3_dmhbJiQ3qHjOb4qDwRUSlrGQynWYNwCysijoyE8wEmUig_aem_ddpdF3McDto0LM9tu2U8NQ) and place it in the `data/` directory.

### Running the Notebook
1. Install Jupyter Notebook:
   ```bash
   pip install notebook
   ```
2. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the provided notebook file and execute the cells sequentially.

## Conclusion
This project demonstrates the effectiveness of CNN models in detecting COVID-19 from chest X-ray images. With an accuracy of **95.65%**, the model offers a rapid and scalable diagnostic tool, especially useful in resource-limited settings. Future work will focus on validating the model on larger datasets and exploring its integration into clinical workflows.
