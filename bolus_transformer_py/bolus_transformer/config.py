"""Hyperparameters and signal/column definitions shared across the pipeline."""
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch

XML_SIGNALS: List[str] = [
    "glucose_level", "basis_heart_rate", "basis_gsr",
    "basis_skin_temperature", "basis_air_temperature",
]

SIGNALS: List[str] = [
    "glucose_level", "basis_heart_rate", "basis_gsr",
    "basis_skin_temperature", "basis_air_temperature",
    "running_IOB", "self_reported_carbs", "sin_t", "cos_t",
]

SIGNALS_FOR_LOSS: List[str] = [
    "glucose_level", "basis_heart_rate", "basis_gsr",
    "basis_skin_temperature", "basis_air_temperature",
]

# Arbitrary reference epoch used to turn timestamps into a "running minutes" axis.
START_TIME = pd.Timestamp("2020-06-01 10:00:00")


@dataclass(frozen=True)
class Config:
    # --- model ---
    n_embed: int = 16
    context_length: int = 36          # 3 hours of 5-minute tokens
    d_model: int = 32
    attention_heads_per_block: int = 4
    num_blocks: int = 3
    dropout: float = 0.1

    # Positions before this offset in each window are dropped from the loss/labels:
    # the causal decoder hasn't seen enough context yet to predict them well.
    eval_offset: int = 12

    # --- training ---
    batch_size: int = 32
    horizon: int = 6                  # 30-minute-ahead pretraining target
    finetune_label_window: int = 4    # bolus label tolerance window width (tokens)
    pretrain_epochs: int = 3000
    finetune_epochs: int = 1500
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4

    # Per-signal pretraining loss weights, in SIGNALS_FOR_LOSS order (glucose weighted 4x).
    loss_weights: Tuple[float, ...] = (1.0, 0.25, 0.25, 0.25, 0.25)

    # Insulin-on-board decay curve parameters (minutes to peak / total duration of action).
    iob_peak: float = 75.0
    iob_dia: float = 300.0

    def weights_tensor(self) -> torch.Tensor:
        return torch.tensor(self.loss_weights)


DEFAULT_CONFIG = Config()
