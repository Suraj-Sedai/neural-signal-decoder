# Neural Signal Decoder (Simulated EEG)

**End‑to‑End Neural Signal Decoding with Frequency‑Aligned Modeling**

---

## 1. Overview

This project implements a complete neural signal decoding pipeline using *simulated EEG data* to study architectural choices for brain–computer interface (BCI) systems. The work emphasizes **engineering rigor, inductive bias alignment, and failure analysis**, rather than raw performance alone.

Multiple modeling paradigms are evaluated—time‑domain sequence models (LSTM, Transformer) and frequency‑domain representations—culminating in a **bandpower‑based Transformer** that reflects established neuroscience priors.

The final system is modular, reproducible, and suitable as a research or portfolio‑grade project.

---

## 2. Project Objectives

### 2.1 Primary Objectives

- Decode multichannel EEG‑like signals into discrete control classes
- Evaluate sequence modeling architectures under controlled conditions
- Study inductive bias mismatch in time‑series learning
- Design a low‑latency, interpretable decoding pipeline

### 2.2 Secondary Objectives

- Maintain strict modular separation
- Enable reproducibility and extensibility
- Provide failure‑driven architectural justification

---

## 3. High‑Level System Architecture

**End‑to‑End Pipeline:**

Simulated EEG → Preprocessing → Windowing → Feature Extraction → Model → Class Prediction

```
┌──────────────────┐
│ Signal Generator │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Preprocessing    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Windowing        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Feature Encoding │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Model Core       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Classifier       │
└──────────────────┘
```

---

## 4. Signal Simulation

### 4.1 Purpose

Generate realistic yet controllable EEG‑like signals with class‑specific spectral signatures.

### 4.2 Signal Properties

- Multichannel sinusoidal signals
- Class‑specific frequency bands:
  - Alpha (8–12 Hz)
  - Beta (13–30 Hz)
  - Theta (4–7 Hz)
  - Gamma (30–45 Hz)
- Channel‑wise phase variation
- Additive Gaussian noise
- Per‑channel normalization

### 4.3 Rationale

This design enforces **frequency‑dominant class separability**, mirroring how EEG is interpreted in real BCIs.

---

## 5. Preprocessing

### 5.1 Normalization

- Z‑score normalization per channel
- Prevents amplitude dominance

### 5.2 Windowing

- Sliding windows over time
- Parameters:
  - Window size
  - Stride

**Trade‑off:**
- Smaller windows → lower latency
- Larger windows → more context

---

## 6. Feature Representations

### 6.1 Time‑Domain Windows

Raw signal windows used for LSTM and Transformer baselines.

### 6.2 Frequency‑Domain (Bandpower)

Final model uses **bandpower features**, computed via FFT:

- Power spectral density per channel
- Aggregation into canonical EEG bands
- Output shape: `(channels × frequency bands)`

**Key Insight:**
> EEG decoding is inherently spectral; models should not be forced to rediscover Fourier structure.

---

## 7. Models Evaluated

### 7.1 LSTM (Time‑Domain)

**Architecture:**
- Multi‑layer LSTM
- Temporal aggregation via final hidden state

**Outcome:**
- Failed to generalize consistently

**Reason:**
- Strong temporal bias misaligned with stationary oscillatory signals

---

### 7.2 Transformer (Time‑Domain)

**Architecture:**
- Self‑attention over timesteps
- Positional embeddings

**Outcome:**
- Mode collapse and class dominance

**Reason:**
- Excess capacity without meaningful temporal semantics

---

### 7.3 Spectral MLP (Baseline)

- FFT‑based features
- Simple feedforward classifier

**Outcome:**
- Near‑perfect accuracy

**Conclusion:**
- Feature representation mattered more than architectural complexity

---

### 7.4 Bandpower Transformer (Final Model)

**Architecture:**
- Tokens represent EEG channels
- Features represent bandpower values
- Transformer encoder models inter‑channel relationships

**Why This Works:**
- Matches neuroscience priors
- Attention models spatial (not temporal) structure
- Minimal inductive bias mismatch

---

## 8. Training Pipeline

- Loss: Cross‑Entropy
- Optimizer: Adam
- Train / Test split (80 / 20)
- Batch‑based training
- Evaluation after convergence

---

## 9. Evaluation Metrics

- Overall accuracy
- Class‑wise accuracy
- Confusion matrix

**Final Bandpower Transformer Results:**

- Test Accuracy: **85%**
- Confusions primarily between adjacent frequency bands

This mirrors real EEG ambiguity.

---

## 10. Failure Analysis Summary

| Model | Result | Root Cause |
|------|--------|-----------|
| LSTM | Failed | Incorrect temporal inductive bias |
| Transformer (time) | Failed | Attention over meaningless tokens |
| Spectral MLP | Succeeded | Correct feature alignment |
| Bandpower Transformer | Succeeded | Structure + interpretability |

---

## 11. Key Takeaways

- Model success depends more on **representation alignment** than depth
- Transformers are powerful but not universally applicable
- Frequency‑domain features are essential for EEG decoding
- Failure analysis is as valuable as performance

---

## 12. Limitations

- Simulated data lacks biological artifacts
- No online / real‑time inference tested

---

## 13. Future Work

- Integration with real EEG datasets
- Online decoding experiments
- Cross‑subject generalization
- Hybrid spatial–temporal attention

---

## 14. Conclusion

This project demonstrates a principled, failure‑aware approach to neural signal decoding. By aligning model architecture with domain structure, the final system achieves strong performance while remaining interpretable and extensible. The work highlights a core lesson of applied machine learning:

> **Choosing the right representation matters more than choosing the most complex model.**

