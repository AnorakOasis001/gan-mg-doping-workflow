from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


class LogSumExpAccumulator:
    def __init__(self) -> None:
        self._m: float | None = None
        self._s: float = 0.0

    def update(self, x: NDArray[np.float64]) -> None:
        if x.size == 0:
            return

        m2 = float(np.max(x))
        s2 = float(np.sum(np.exp(x - m2)))

        if self._m is None:
            self._m = m2
            self._s = s2
            return

        if m2 <= self._m:
            self._s = self._s + math.exp(m2 - self._m) * s2
        else:
            self._s = math.exp(self._m - m2) * self._s + s2
            self._m = m2

    def logsumexp(self) -> float:
        if self._m is None:
            raise ValueError("No values were provided to LogSumExpAccumulator.")
        return self._m + math.log(self._s)


class RunningStats:
    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._min: float | None = None

    def update(self, values: NDArray[np.float64]) -> None:
        if values.size == 0:
            return

        for value in values:
            value_f = float(value)
            self._count += 1
            delta = value_f - self._mean
            self._mean += delta / self._count
            if self._min is None or value_f < self._min:
                self._min = value_f

    @property
    def count(self) -> int:
        if self._count == 0:
            raise ValueError("RunningStats.count is undefined before any updates.")
        return self._count

    @property
    def mean(self) -> float:
        if self._count == 0:
            raise ValueError("RunningStats.mean is undefined before any updates.")
        return self._mean

    @property
    def min(self) -> float:
        if self._min is None:
            raise ValueError("RunningStats.min is undefined before any updates.")
        return self._min


class ScaledExpSumAccumulator:
    """Accumulate sums of the form Σ a_i * exp(x_i) in a stable scaled representation."""

    def __init__(self) -> None:
        self._m: float | None = None
        self._s: float = 0.0

    def update(self, x: NDArray[np.float64], a: NDArray[np.float64]) -> None:
        if x.size == 0:
            return
        if x.shape != a.shape:
            raise ValueError("x and a must have identical shapes.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")
        if not np.all(np.isfinite(a)):
            raise ValueError("a must contain only finite values.")
        if np.any(a < 0.0):
            raise ValueError("a must be non-negative.")

        mask = a > 0.0
        if not np.any(mask):
            return

        x_pos = x[mask]
        a_pos = a[mask]

        m2 = float(np.max(x_pos))
        s2 = float(np.sum(a_pos * np.exp(x_pos - m2)))

        if self._m is None:
            self._m = m2
            self._s = s2
            return

        if m2 <= self._m:
            self._s = self._s + math.exp(m2 - self._m) * s2
        else:
            self._s = math.exp(self._m - m2) * self._s + s2
            self._m = m2

    def is_zero(self) -> bool:
        return self._m is None or self._s == 0.0

    def value_log(self) -> float:
        if self.is_zero():
            return float("-inf")
        assert self._m is not None
        return self._m + math.log(self._s)
