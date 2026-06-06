# Feedforward Neural Network for MNIST Digit Classification

A complete implementation of a Feedforward Neural Network (FFNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

## Overview

**Dataset:** MNIST (Modified National Institute of Standards and Technology)

- Total samples: 70,000 (60,000 training + 10,000 test)
- Image size: 28×28 pixels (grayscale)
- Classes: 10 (digits 0-9)

## Network Architecture

The model is a Feedforward Neural Network (FFNN) designed to classify 28×28 flattened MNIST images (784 inputs) into 10 digit classes.

### Architecture Layout

```txt
Input (784 features)
    ↓
Linear (784 → 128) → BatchNorm → ReLU → Dropout (0.2)
    ↓
Linear (128 → 64)  → BatchNorm → ReLU → Dropout (0.2)
    ↓
Linear (64 → 32)   → BatchNorm → ReLU → Dropout (0.2)
    ↓
Linear (32 → 10)   → Output (Logits for Cross-Entropy Loss)
```

### Components and Rationale

- **Input Layer (784):** Accepts flattened 28×28 grayscale images standardized to Mean = 0.1307, Std = 0.3081.
- **Hidden Layers (128 → 64 → 32):**
  - **Linear (`nn.Linear`):** Maps features to lower-dimensional spaces.
  - **Batch Normalization (`nn.BatchNorm1d`):** Stabilizes and accelerates training by normalizing activations across the batch.
  - **ReLU Activation:** Introduces non-linearity ($f(x) = \max(0, x)$) and avoids vanishing gradients.
  - **Dropout (0.2):** Randomly deactivates 20% of neurons during training to prevent overfitting.
- **Output Layer (10):** Maps the final 32 hidden units to 10 class logits. During training, PyTorch's `nn.CrossEntropyLoss` internally applies LogSoftmax.

### Design Rationale

- **Progressive Dimensionality Reduction:** The transition of $784 \to 128 \to 64 \to 32 \to 10$ acts as a feature bottleneck to prevent overfitting.
- **Regularization & Normalization:** Combining Batch Normalization and Dropout forces the network to learn more robust, redundant representations and improves test set generalization (>97% accuracy).

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
