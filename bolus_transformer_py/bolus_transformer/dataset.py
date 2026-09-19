"""Random-window batch samplers for pretraining and fine-tuning.

These replace the notebook's module-level globals (`training_d`, `testing_d`,
`bolus_train`, `bolus_test`) with small stateful objects so a fold's data doesn't
leak into the next one.
"""
from typing import Dict, Tuple

import torch


class PretrainBatcher:
    """Samples (context, target, mask) windows for the next-timestep pretraining
    objective. Each of `train_data`/`test_data` maps subject_id -> (features,
    targets, mask) numpy arrays, all sharing the same length per subject."""

    def __init__(self, train_data: Dict, test_data: Dict, context_length: int, horizon: int):
        self.train_data = train_data
        self.test_data = test_data
        self.context_length = context_length
        self.horizon = horizon

    def _data_for(self, split: str) -> Dict:
        return self.train_data if split == "train" else self.test_data

    def get_batch(self, batch_size: int, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_lib = self._data_for(split)
        key_list = list(data_lib.keys())
        subject_idx = torch.randint(len(data_lib), (batch_size,))
        c, h = self.context_length, self.horizon
        ids = [
            (key_list[i], torch.randint(len(data_lib[key_list[i]][0]) - c - h, (1,)).item())
            for i in subject_idx
        ]

        bx = torch.stack([torch.from_numpy(data_lib[sid][0][i:i + c]) for sid, i in ids])
        by = torch.stack([torch.from_numpy(data_lib[sid][1][i + h:i + c + h]) for sid, i in ids])
        bm = torch.stack([torch.from_numpy(data_lib[sid][2][i + h:i + c + h]) for sid, i in ids])
        return bx, by, bm


class FinetuneBatcher:
    """Samples (context, bolus-label) windows for the fine-tuning classification
    head. Each of `train_data`/`test_data` maps subject_id -> (features,
    bolus_labels) numpy arrays."""

    def __init__(self, train_data: Dict, test_data: Dict, context_length: int):
        self.train_data = train_data
        self.test_data = test_data
        self.context_length = context_length

    def _data_for(self, split: str) -> Dict:
        return self.train_data if split == "train" else self.test_data

    def get_batch(self, batch_size: int, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data_lib = self._data_for(split)
        key_list = list(data_lib.keys())
        subject_idx = torch.randint(len(data_lib), (batch_size,))
        c = self.context_length
        ids = [
            (key_list[i], torch.randint(len(data_lib[key_list[i]][0]) - c, (1,)).item())
            for i in subject_idx
        ]

        bx = torch.stack([torch.from_numpy(data_lib[sid][0][i:i + c]) for sid, i in ids])
        by = torch.stack([torch.from_numpy(data_lib[sid][1][i:i + c]) for sid, i in ids])
        return bx, by.unsqueeze(-1)
