from __future__ import annotations
import numpy as np
import pandas as pd

def erc_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n)/n
    for _ in range(max_iter):
        cw = cov @ w
        rc = w * cw
        total = rc.sum()
        if total <= 1e-16:
            break
        target = total / n
        w_new = w * (target / (rc + 1e-12))
        w_new = np.clip(w_new, 0.0, None)
        s = w_new.sum()
        w_new = w_new / s if s > 0 else np.ones(n)/n
        if np.linalg.norm(w_new - w, 1) < tol:
            w = w_new
            break
        w = w_new
    return w

def target_leverage(weights: np.ndarray, cov: np.ndarray, target_vol_ann: float, periods_per_year: int = 252) -> float:
    vol_daily = float(np.sqrt(weights @ cov @ weights))
    vol_ann = vol_daily * np.sqrt(periods_per_year)
    if vol_ann <= 0:
        return 0.0
    return target_vol_ann / vol_ann
