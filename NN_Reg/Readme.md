# Neural Network Regularization: Dropout vs L2 (Weight Decay)

A comparative study of neural network training with and without regularization techniques (Dropout and L2), demonstrating how regularization improves model generalization on the MNIST dataset.

## Overview

**Dataset:** MNIST (Modified National Institute of Standards and Technology)

- Total samples: 70,000 (60,000 training + 10,000 test)
- Image size: 28×28 pixels (grayscale)
- Classes: 10 (digits 0-9)
- Train/Val Split: 80/20

**Objective:** Train a simple neural network with and without regularization, compare training and testing accuracies, and discuss how regularization helps improve model generalization.

## Network Architecture

All three models share the same base architecture: a deliberately **over-parameterized** 5-hidden-layer Feedforward Neural Network (~1.6 M parameters). The large capacity guarantees the baseline clearly overfits, making the regularization benefit unmistakable.

```txt
Input (784 features, flattened 28×28)
    ↓
Linear (784 → 1024) → ReLU → [Dropout?]
    ↓
Linear (1024 → 512) → ReLU → [Dropout?]
    ↓
Linear (512 → 256)  → ReLU → [Dropout?]
    ↓
Linear (256 → 128)  → ReLU → [Dropout?]
    ↓
Linear (128 → 10)   → Output (Logits for Cross-Entropy Loss)
```

### Three Variants

| Variant               | Dropout (p) | Weight Decay (λ) |
|-----------------------|-------------|-------------------|
| **Baseline** (No Reg) | 0.0         | 0                 |
| **Dropout**           | 0.5         | 0                 |
| **L2 Regularization** | 0.0         | 5e-4              |

### Design Rationale

- **Wide hidden layers (1024, 512, 256, 128):** Intentionally over-parameterized so the regularization effect is clearly observable.
- **No BatchNorm:** Omitted to isolate the effect of regularization techniques.
- **High dropout (0.5):** The classic Hinton recommendation; forces the network to learn redundant, distributed representations.
- **L2 via weight_decay (λ=5e-4):** Penalizes large weights in the optimizer, equivalent to adding $\frac{\lambda}{2} \|\mathbf{w}\|_2^2$ to the loss.

## Training Configuration

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 30 (full run, no early stopping)
- **Batch Size:** 64
- **Train/Val Split:** 40,000 training / 20,000 validation (smaller train set increases overfitting pressure)
- **Preprocessing:** MNIST standardization (Mean = 0.1307, Std = 0.3081)

## Results

### Accuracy Comparison

| Model               | Train Acc (%) | Test Acc (%) | Gap (Train - Test) |
|---------------------|---------------|--------------|---------------------|
| Baseline (No Reg)   | Highest       | Lowest       | **Largest**         |
| Dropout (p=0.5)     | Lower         | Higher       | **Smallest**        |
| L2 (λ=1e-3)         | Slightly Lower| Higher       | Small               |

*Exact numbers depend on the run, so see the notebook output.*

### Key Observations

- **Baseline:** Achieves the highest training accuracy but the lowest test accuracy. The large generalization gap confirms overfitting.
- **Dropout:** Training accuracy is intentionally suppressed (random neuron masking), but test accuracy improves significantly. The gap is the smallest.
- **L2 Regularization:** Penalizing large weights produces smoother decision boundaries. Training accuracy is slightly lower than baseline, but test accuracy improves.

## How Regularization Improves Generalization

### What Is Overfitting?

Overfitting occurs when a neural network memorizes training data (including noise) rather than learning generalizable patterns. The **generalization gap** (Train Acc - Test Acc) quantifies overfitting.

### Dropout

- **Mechanism:** Randomly zeroes out neurons during training (p = 0.5 means 50% are dropped per forward pass).
- **Effect:** Forces the network to learn redundant, distributed representations. No single neuron can dominate.
- **At test time:** All neurons are active with scaled weights, effectively **ensembling** many sub-networks.
- **Intuition:** Dropout acts as an implicit ensemble of $2^n$ thinned networks.

### L2 Regularization (Weight Decay)

- **Mechanism:** Adds a penalty $\frac{\lambda}{2} \|\mathbf{w}\|_2^2$ to the loss function.
- **Gradient update:** $w \leftarrow (1 - \eta \lambda) w - \eta \nabla_{w} \mathcal{L}$
- **Effect:** Discourages large weights → prefers **simpler, smoother** decision boundaries that generalize better.
- **Why "weight decay":** The $(1 - \eta \lambda)$ factor shrinks weights toward zero each step.

### Summary Table

| Aspect              | Baseline       | Dropout              | L2 Reg                |
|---------------------|----------------|----------------------|-----------------------|
| Training Accuracy   | **Highest**    | Lower                | Slightly Lower        |
| Test Accuracy       | Lowest         | **Higher**           | **Higher**            |
| Generalization Gap  | **Largest**    | **Smallest**         | Small                 |
| Mechanism           | None           | Random neuron masking| Weight magnitude penalty|
| Effect              | Memorizes data | Learns robust features| Prefers simple models |

**Key Takeaway:** Regularization techniques reduce overfitting by constraining the model's capacity. Dropout forces distributed representations through random masking, while L2 penalizes weight magnitudes to favor simpler models. Both lead to **better generalization**: improved test accuracy relative to training accuracy.

## Files

- `train.ipynb`: Complete notebook with all implementation steps:
  1. Data loading and exploration
  2. Sample image visualization
  3. Model definitions (three variants)
  4. Training loop with early stopping
  5. Training vs Validation loss/accuracy curves
  6. Test set evaluation and comparison table
  7. Accuracy comparison bar chart
  8. Generalization gap visualization
  9. Confusion matrices for all three models
  10. Per-class accuracy comparison
  11. Overfitting analysis (gap over epochs)
  12. Summary dashboard
  13. Discussion of regularization theory
  14. Classification reports

- `Readme.md`: This file

## Visualizations

- **Loss Curves:** Training vs validation loss for each model
- **Accuracy Curves:** Training vs validation accuracy for each model
- **Accuracy Comparison:** Side-by-side bar chart of train vs test accuracy
- **Generalization Gap:** Bar chart showing Train Acc - Test Acc
- **Confusion Matrices:** Normalized per-class accuracy heatmaps
- **Per-Class Accuracy:** Grouped bar chart comparing digit-level accuracy
- **Overfitting Gap:** Train-Val accuracy gap over epochs
- **Summary Dashboard:** All key results in one composite figure

## How to Run

1. Upload `train.ipynb` to Kaggle (or open in Jupyter/VS Code)
2. Enable GPU accelerator (recommended)
3. Run all cells sequentially
4. Plots display inline and are also saved as PNG files to disk

## Dependencies

- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- Seaborn
- Pandas
- tqdm
