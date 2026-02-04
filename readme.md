# Neural Signal Decoder (Simulated EEG)

**A principled study of neural signal decoding using frequency-aligned modeling**

---

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [System Architecture](#system-architecture)
- [Signal Simulation](#signal-simulation)
- [Preprocessing](#preprocessing)
- [Feature Representations](#feature-representations)
- [Models Evaluated](#models-evaluated)
  - [LSTM (Time-Domain)](#lstm-time-domain)
  - [Transformer (Time-Domain)](#transformer-time-domain)
  - [Spectral MLP (Baseline)](#spectral-mlp-baseline)
  - [Bandpower Transformer (Final Model)](#bandpower-transformer-final-model)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Failure Analysis](#failure-analysis)
- [Key Takeaways](#key-takeaways)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Conclusion](#conclusion)

---

## Overview

This repository provides an end-to-end neural signal decoding pipeline using **simulated EEG data**. The study emphasizes model architecture evaluation, failure analysis, and alignment of inductive bias with signal structure. The final model utilizes **bandpower features and a Transformer architecture** that is physiologically meaningful and interpretable.

---

## Project Objectives

**Primary Objectives:**

- Decode multichannel EEG-like signals into discrete classes
- Compare LSTM, Transformer, spectral MLP, and bandpower-based Transformer
- Analyze failures due to inductive bias mismatch

**Secondary Objectives:**

- Modular, reproducible code structure
- Clear documentation for research and portfolio purposes

---

## System Architecture

```flow
st=>start: Simulated EEG Signal
pre=>operation: Preprocessing (Normalization)
win=>operation: Sliding Window Segmentation
feat=>operation: Feature Extraction (Time / FFT / Bandpower)
model=>operation: Model Training / Inference
class=>end: Class Prediction

st->pre->win->feat->model->class
```

The system pipeline ensures reproducibility and clean separation of components.

---

## Signal Simulation

- Multi-channel sinusoidal EEG-like signals
- Class-specific frequency bands:
  - Theta (4–7 Hz)
  - Alpha (8–12 Hz)
  - Beta (13–30 Hz)
  - Gamma (30–45 Hz)
- Random phase per channel
- Additive Gaussian noise
- Channel-wise normalization

---

## Preprocessing

- **Normalization:** Z-score per channel
- **Windowing:** Sliding windows with configurable size and stride

---

## Feature Representations

- **Time-Domain Windows**: For LSTM and Time-Domain Transformer
- **FFT Spectral Features**: For spectral MLP
- **Bandpower Features**: Final physiologically-aligned representation

---

## Models Evaluated

### LSTM (Time-Domain)

- Failed to generalize due to incorrect temporal inductive bias

### Transformer (Time-Domain)

- Mode collapse due to meaningless positional encoding

### Spectral MLP (Baseline)

- Achieved near-perfect accuracy with FFT features
- Shows representation matters more than complexity

### Bandpower Transformer (Final Model)

- Tokens represent channels, features represent bandpower
- Self-attention models inter-channel relationships
- Achieved 85% test accuracy with interpretable results

---

## Training Pipeline

- Cross-Entropy Loss
- Adam Optimizer
- 80/20 Train/Test split
- Batch-based training with evaluation

---

## Evaluation Metrics

- Overall accuracy
- Class-wise accuracy
- Confusion matrix

---

## Results

**Bandpower Transformer Test Accuracy:** 85%

Confusion primarily between adjacent bands (Theta/Alpha, Beta/Gamma) indicating realistic spectral overlap.

---

## Failure Analysis

| Model | Result | Root Cause |
|-------|--------|------------|
| LSTM | Failed | Temporal inductive bias mismatch |
| Transformer | Failed | Attention over meaningless time steps |
| Spectral MLP | Succeeded | Correct feature alignment |
| Bandpower Transformer | Succeeded | Structure + interpretability |

---

## Key Takeaways

- Representation alignment is more critical than architecture complexity
- Sequence models are not universally applicable
- Bandpower features are essential for EEG decoding
- Attention models are effective when applied to meaningful tokens

---

## Limitations

- Simulated signals, no real physiological artifacts
- Offline evaluation only
- Single-subject simulation

---

## Future Work

- Apply to real EEG datasets
- Real-time decoding experiments
- Cross-subject generalization
- Hybrid spatial-temporal attention models

---

## Repository Structure

```
notebooks/          # Experimentation notebooks
docs/               # Project documentation
src/                # Source code
  data/             # Signal simulation scripts
  utils/            # Preprocessing utilities
  models/           # All neural models
  training/         # Training & evaluation scripts
results/            # Confusion matrices and plots
requirements.txt    # Required Python packages
README.md           # This file
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

---

## Running the Project

The main experimentation workflow is in `notebooks/Neural_Signal_Decoding_Simulated_EEG.ipynb`.

It includes:
- Signal visualization
- Model training
- Evaluation and confusion matrices
- Bandpower Transformer final evaluation

---

## Conclusion

This project demonstrates that **choosing the correct representation matters more than choosing a complex model**. The Bandpower Transformer aligns with EEG spectral characteristics, achieves strong performance, and provides interpretable outputs.

---

## License

Educational and research purposes only.

