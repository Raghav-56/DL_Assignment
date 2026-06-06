# Feedforward Neural Network for MNIST Digit Classification

A complete implementation of a Feedforward Neural Network (FFNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

## Overview

**Dataset:** MNIST (Modified National Institute of Standards and Technology)
- Total samples: 70,000 (60,000 training + 10,000 test)
- Image size: 28×28 pixels (grayscale)
- Classes: 10 (digits 0-9)

## Network Architecture

```
Input Layer (784 neurons: 28×28 flattened)
    ↓
Hidden Layer 1 (128 neurons + ReLU + BatchNorm + Dropout 0.2)
    ↓
Hidden Layer 2 (64 neurons + ReLU + BatchNorm + Dropout 0.2)
    ↓
Hidden Layer 3 (32 neurons + ReLU + BatchNorm + Dropout 0.2)
    ↓
Output Layer (10 neurons + Softmax)
```

### Architecture Rationale
- **Progressive dimensionality reduction:** 784 → 128 → 64 → 32 → 10 prevents overfitting
- **ReLU activation:** Non-linear activation, avoids vanishing gradient problem
- **Batch Normalization:** Stabilizes training, accelerates convergence
- **Dropout (0.2):** Regularization technique to reduce overfitting
- **CrossEntropyLoss:** Ideal for multi-class classification

## Training Configuration

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 15 (with early stopping if validation loss plateaus)
- **Batch Size:** 64
- **Early Stopping Patience:** 3 epochs
- **Train/Val Split:** 80/20

## Results

- **Test Accuracy:** >97%
- **Model Parameters:** ~130,000
- **Per-class Accuracy:** Individual accuracy per digit (0-9)

## Files

- `train.ipynb` — Complete notebook with all implementation steps:
  1. Data loading and exploration
  2. Preprocessing (normalization, flattening)
  3. Model definition and configuration
  4. Training loop with validation
  5. Visualization of training curves
  6. Test set evaluation
  7. Confusion matrix and error analysis
  8. Model saving

- `README.md` — This file

## Key Implementation Highlights

### Data Preprocessing
- Normalization: Pixel values normalized to [0,1] range
- Standardization: Mean = 0.1307, Std = 0.3081
- Flattening: 28×28 images → 784-dimensional vectors

### Training Loop
- Forward pass → Loss computation → Backward pass → Parameter update
- Validation after each epoch to monitor overfitting
- Early stopping to prevent unnecessary training

### Visualization
- Training vs Validation loss and accuracy curves
- Sample predictions (correct and incorrect)
- Confusion matrix (raw and normalized)
- Error analysis with misclassified examples

## Performance Metrics

- **Accuracy:** Percentage of correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **Per-Class Accuracy:** Individual accuracy for each digit

## How to Run

1. Open `train.ipynb` in Jupyter Notebook or JupyterLab
2. Execute cells sequentially (or use "Run All")
3. Training will start automatically and display progress
4. View results, confusion matrix, and error analysis
5. Trained model saved as `mnist_ffnn_model.pth`

## Dependencies

- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- Seaborn
- Pandas

## References & Resources

See `links.txt` and `links_ai.txt` for comprehensive learning resources including:
- PyTorch tutorials and documentation
- Neural network theory and backpropagation explanations
- MNIST dataset information
- Machine learning best practices
