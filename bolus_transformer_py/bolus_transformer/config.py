from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch

XML_SIGNALS: List[str] = [
    "glucose_level", "basis_heart_rate", "basis_gsr",
    "basis_skin_temperature", "basis_air_temperature",
]
WEARABLE_SIGNALS: List[str] = XML_SIGNALS[1:]
SIGNALS: List[str] = XML_SIGNALS + ["running_IOB", "self_reported_carbs", "sin_t", "cos_t"]
SIGNALS_FOR_LOSS: List[str] = XML_SIGNALS
NORMALIZED_SIGNALS: List[str] = SIGNALS[:-2]
LOG_SIGNALS: List[str] = ["basis_gsr", "self_reported_carbs"]

START_TIME = pd.Timestamp("2020-06-01 10:00:00")


@dataclass(frozen=True)
class Config:
    n_embed: int = 16
    context_length: int = 36
    d_model: int = 32
    attention_heads: int = 4
    num_blocks: int = 3
    dropout: float = 0.1
    eval_offset: int = 12

    horizon: int = 6
    label_window_minutes: float = 20.0

    batch_size: int = 32
    pretrain_steps: int = 3000
    finetune_steps: int = 1500
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    eval_every: int = 100
    patience: int = 10

    loss_weights: Tuple[float, ...] = (1.0, 0.25, 0.25, 0.25, 0.25)

    iob_peak: float = 75.0
    iob_dia: float = 300.0

    target_false_alarms_per_day: float = 4.0

    def weights_tensor(self) -> torch.Tensor:
        return torch.tensor(self.loss_weights, dtype=torch.float32)


DEFAULT_CONFIG = Config()
