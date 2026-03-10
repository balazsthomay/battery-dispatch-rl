"""Do-nothing baseline -- always idles."""

from __future__ import annotations

import numpy as np


class DoNothingPolicy:
    """Always returns action=0 (idle).

    Provides a lower-bound baseline: the battery does nothing,
    incurring only calendar degradation with zero revenue.
    """

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Match stable-baselines3 predict() interface.

        Args:
            obs: Observation array, shape (32,) or (batch, 32).
            deterministic: Ignored (always deterministic).

        Returns:
            (action, None) where action is zeros with appropriate shape.
        """
        if obs.ndim == 1:
            return np.array([0.0], dtype=np.float32), None
        # Batched
        return np.zeros((obs.shape[0], 1), dtype=np.float32), None
