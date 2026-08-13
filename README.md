# bolus-transformer

A transformer-based model for predicting **when** a type 1 diabetes patient should take a bolus insulin dose, trained on continuous glucose monitoring (CGM) and physiological sensor data from the OhioT1DM dataset.

## Overview

This project applies a causal transformer encoder to multimodal physiological signals—glucose, heart rate, galvanic skin response (GSR), skin temperature, air temperature, and insulin-on-board—to predict optimal bolus insulin timing. Unlike traditional models that estimate *dosage*, this approach addresses the timing prediction problem: given current glucose trends and physiological state, should the patient take a bolus now?

## Dataset

- **OhioT1DM**: 12 subjects (~162,000 timesteps across the cohort, ~2,230 logged bolus events)
- **Signals**: Continuous glucose (CGM), heart rate, GSR, skin temperature, air temperature, meal/carb logs
- **Device cohorts**: 
  - 2018 cohort (6 subjects): Basis sensor band
  - 2020 cohort (6 subjects): Empatica watch

## Model Architecture

### Pretraining (Unsupervised)
- **Architecture**: Causal transformer encoder (GPT-style)
- **Context window**: 3 hours (36 tokens @ 5-minute resolution)
- **Tokenization**: Per-signal linear projections into embedding space, concatenated with meal-type embeddings, carb-amount projections, and cyclic (sin/cos) time-of-day encodings
- **Objective**: Masked MSE on next-timestep prediction (causal), handling missing values per-signal

### Fine-tuning (Supervised Classification)
- **Head**: Binary classification on the encoder's final-timestep representation
- **Loss**: Class-weighted binary cross-entropy
- **Target**: Whether a bolus should be given (tolerance window around logged bolus timestamps)

## Key Results

6-fold leave-one-subject-out (LOSO) cross-validation:
- **Pretraining**: ~17% MSE reduction over persistence baseline across all folds
- **Bolus timing (test set)**:
  - Mean AUROC: **0.760** (std 0.055)
  - Mean AP: **0.328** (std 0.046)
  - Base rate: ~6–7% (highly imbalanced)
  - Consistent performance across subjects

## Preprocessing

- **Normalization**: Z-score per signal, computed from pooled mean/std across training subjects (per LOSO fold)
- **GSR**: Log-transformed; glucose left linear
- **Insulin-on-board**: Deterministic feature via predefined decay curve from dose logs
- **Alignment**: Overlap-trimmed `merge_asof` for timestamp synchronization

## Files

- `bolus_insulin_model.ipynb`: Full pipeline (pretraining, fine-tuning, LOSO evaluation)
- `README.md`: This file

## Getting Started

1. Obtain the OhioT1DM dataset and place it in the expected directory
2. Run the notebook end-to-end for pretraining and cross-validation
3. Inspect results and per-subject performance curves

## References

- **OhioT1DM**: Marling et al. (2018) – public benchmark for type 1 diabetes modeling
- **Model inspiration**: Autoregressive transformers for medical time series

## License

MIT

---

**Note**: This project is research-focused. Any clinical application would require rigorous validation and regulatory review.
