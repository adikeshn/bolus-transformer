from typing import Dict, List, Tuple

import numpy as np
import torch


class WindowSampler:
    def __init__(self, arrays: Dict[str, Dict[str, np.ndarray]], context_length: int, horizon: int = 0):
        self.context_length = context_length
        self.horizon = horizon
        need = context_length + horizon + 1
        self.arrays = [a for a in arrays.values() if len(a["x"]) >= need]
        if not self.arrays:
            raise ValueError("No sequence is long enough for one training window.")

    def _starts(self, batch_size: int) -> List[Tuple[dict, int]]:
        picks = torch.randint(len(self.arrays), (batch_size,)).tolist()
        out = []
        for p in picks:
            a = self.arrays[p]
            hi = len(a["x"]) - self.context_length - self.horizon
            out.append((a, torch.randint(hi, (1,)).item()))
        return out

    def pretrain_batch(self, batch_size: int):
        c, h = self.context_length, self.horizon
        starts = self._starts(batch_size)
        x = np.stack([a["x"][i:i + c] for a, i in starts])
        y = np.stack([a["y"][i + h:i + c + h] for a, i in starts])
        m = np.stack([a["mask"][i + h:i + c + h] for a, i in starts])
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)

    def finetune_batch(self, batch_size: int):
        c = self.context_length
        starts = self._starts(batch_size)
        x = np.stack([a["x"][i:i + c] for a, i in starts])
        y = np.stack([a["labels"][i:i + c] for a, i in starts])
        return torch.from_numpy(x), torch.from_numpy(y)


def sweep_windows(n: int, context_length: int, offset: int) -> List[Tuple[int, int]]:
    c = context_length
    if n < c:
        return []
    starts = list(range(0, n - c + 1, c - offset))
    if starts[-1] + c < n:
        starts.append(n - c)
    windows, covered = [], offset
    for s in starts:
        windows.append((s, max(offset, covered - s)))
        covered = s + c
    return windows
