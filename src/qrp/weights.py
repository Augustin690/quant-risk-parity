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


def sp3_weights(
    cov: np.ndarray,
    asset_class_indices: dict[str, list[int]],
    target_vol_ann: float = 0.10,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    S&P Risk Parity Index 3-step weight computation.

    Based on "Indexing Risk Parity Strategies" (S&P Dow Jones Indices, 2018).

    Step 1: Inverse-vol weight each instrument to target vol.
    Step 2: Apply asset-class multiplier so each class targets the same vol.
    Step 3: Apply portfolio multiplier to hit overall target vol.

    Args:
        cov: Annualized covariance matrix (n x n).
        asset_class_indices: Dict mapping class name to list of integer indices
            e.g. {'Equity': [0,1,2], 'Fixed Income': [3,4,5,6], 'Commodities': [7,8,9]}
        target_vol_ann: Target annualized volatility (default 0.10 = 10%).
        periods_per_year: Trading days per year for annualization.

    Returns:
        Array of portfolio weights (may sum to > 1 due to leverage).
    """
    n = cov.shape[0]
    # Individual instrument annualized vols from diagonal of covariance
    instrument_vols = np.sqrt(np.diag(cov))

    # Step 1: inverse-vol raw weights
    raw_weights = np.where(instrument_vols > 0, target_vol_ann / instrument_vols, 0.0)

    # Step 2: asset-class multiplier
    ac_weights = np.zeros(n)
    for ac_name, indices in asset_class_indices.items():
        n_inst = len(indices)
        # Equal-weight the inverse-vol-adjusted instruments within class
        w_ac = np.zeros(n)
        for idx in indices:
            w_ac[idx] = raw_weights[idx] / n_inst

        # Compute asset class sub-portfolio vol
        ac_vol = np.sqrt(w_ac @ cov @ w_ac)
        multiplier = target_vol_ann / ac_vol if ac_vol > 0 else 1.0

        for idx in indices:
            ac_weights[idx] = w_ac[idx] * multiplier

    # Step 3: portfolio multiplier — combine classes equally (1/N_classes)
    n_classes = len(asset_class_indices)
    port_weights = ac_weights / n_classes
    port_vol = np.sqrt(port_weights @ cov @ port_weights)
    port_multiplier = target_vol_ann / port_vol if port_vol > 0 else 1.0

    final_weights = port_weights * port_multiplier
    return final_weights


def compute_risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Compute marginal risk contribution for each asset.

    RC_i = w_i * (Cov @ w)_i / sigma_portfolio

    Returns array of risk contributions that sum to portfolio vol.
    """
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol <= 0:
        return np.zeros_like(weights)
    marginal = weights * (cov @ weights) / port_vol
    return marginal
