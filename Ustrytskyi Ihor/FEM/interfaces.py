from __future__ import annotations
from typing import Protocol
import numpy as np


class PostProcessor(Protocol):
    def stresses_at(self, points: np.ndarray) -> np.ndarray: ...
    def displacements_at(self, points: np.ndarray) -> np.ndarray: ...
