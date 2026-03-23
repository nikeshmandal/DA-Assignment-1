# DA6401 – Assignment 1

## Multi-Layer Perceptron for Image Classification (NumPy Implementation)

**Name:** Nikesh Kumar Mandal
**Roll Number:** ID25M805
**Course:** DA6401 – Introduction to Deep Learning
**Institute:** IIT Madras

---

# Project Overview

This project implements a **fully configurable Multi-Layer Perceptron (MLP)** using **NumPy only** for image classification on the **MNIST** and **Fashion-MNIST** datasets.

The implementation includes the **complete training pipeline**, including:

* Forward propagation
* Backpropagation
* Multiple optimization algorithms
* Configurable neural network architecture
* Command-line interface for training
* Model inference and evaluation

No deep learning frameworks (PyTorch, TensorFlow, JAX) were used.

---

# Implemented Features

### Neural Network

* Fully connected dense layers
* Configurable number of hidden layers
* Configurable neurons per layer

### Activation Functions

* ReLU
* Sigmoid
* Tanh

### Loss Functions

* Cross Entropy
* Mean Squared Error

### Optimizers

* SGD
* Momentum
* NAG
* RMSProp
* Adam
* Nadam

### Weight Initialization

* Random initialization
* Xavier initialization

### Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

# Project Structure

```
Assignment1_ID25M805/
│
├── ann/
│   ├── neural_layer.py
│   ├── activations.py
│   ├── losses.py
│   ├── neural_network.py
│   └── optimizers.py
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── best_model.npy
│   └── best_config.json
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── serialization.py
│
├── log_data.py
├── sweep.yaml
├── requirements.txt
└── README.md
```

---

# Installation

Create a Python environment and install dependencies:

```
pip install numpy scikit-learn matplotlib wandb
```

---

# Training the Model

Example command:

```
python src/train.py \
--dataset mnist \
--epochs 15 \
--batch_size 64 \
--learning_rate 0.001 \
--optimizer adam \
--num_layers 3 \
--hidden_size 128 128 64 \
--activation relu \
--loss cross_entropy \
--weight_init xavier
```

---

# Running Inference

```
python src/inference.py --dataset mnist
```

This will output:

```
Accuracy
Precision
Recall
F1-score
```

---

# Best Model Performance

| Dataset | Accuracy | F1 Score |
| ------- | -------- | -------- |
| MNIST   | ~97%     | ~0.97    |

---

# Reproducibility

The best model weights are stored in:

```
src/best_model.npy
```

and the corresponding configuration is saved in:

```
src/best_config.json
```

---

# Notes

* Implementation uses **NumPy only** as required.
* Dataset loading is done using **scikit-learn utilities**.
* The code is modular to ensure compatibility with the **automated grading pipeline**.

---

# Author

**Nikesh Kumar Mandal**
Roll Number: **ID25M805**
Report link:
https://wandb.ai/id25m805-iitmaana/uncategorized/reports/Report--VmlldzoxNjEzMzQyNA