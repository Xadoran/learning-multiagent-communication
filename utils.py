from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = []
    ret = 0.0
    for r in reversed(rewards):
        ret = r + gamma * ret
        G.append(ret)
    G.reverse()
    return torch.tensor(G, dtype=torch.float32)


def moving_average(x: List[float], window: int = 50) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    cumsum = np.cumsum(np.insert(np.array(x, dtype=np.float32), 0, 0))
    window = min(window, len(x))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_rewards(curves: List[Tuple[str, List[float]]], out_path: str, ma_window: int = 50):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    for label, rewards in curves:
        ma = moving_average(rewards, window=ma_window)
        xs = np.arange(len(ma))
        plt.plot(xs, ma, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Return (moving avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

