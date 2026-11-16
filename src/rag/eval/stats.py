"""Bootstrap confidence intervals.

Resample the per-question values with replacement many times, take the mean of each
resample, and report the 2.5th and 97.5th percentiles of those means. No assumption
about the distribution is needed, which suits 0/1 outcomes on a few dozen questions.
A fixed seed makes the interval reproducible. With few questions the interval is
wide, which is the honest message.
"""

import numpy as np

N_RESAMPLES = 1000
SEED = 0


def bootstrap_ci(
    values: list[float], n_resamples: int = N_RESAMPLES, seed: int = SEED, level: float = 0.95
) -> dict:
    """{"mean", "low", "high", "n"} for the mean of `values` (None fields if empty)."""
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(n_resamples, len(data)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return {"mean": float(data.mean()), "low": float(low), "high": float(high), "n": len(data)}
