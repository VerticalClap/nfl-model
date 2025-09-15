import numpy as np

def logistic(x: float | np.ndarray) -> np.ndarray:
    """Numerically stable logistic (sigmoid)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))
