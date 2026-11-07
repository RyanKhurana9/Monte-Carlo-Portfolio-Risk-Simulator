import numpy as np
from scipy.stats import norm


def var(pnl, alpha):
    """Loss threshold exceeded with probability (1 - alpha). Positive = loss."""
    return -np.quantile(pnl, 1 - alpha)


def cvar(pnl, alpha):
    """Expected loss given loss >= VaR (Expected Shortfall)."""
    threshold = -var(pnl, alpha)
    return -pnl[pnl <= threshold].mean()


def max_drawdown(paths):
    """Per-path maximum peak-to-trough decline (fraction)."""
    peak = np.maximum.accumulate(paths, axis=1)
    return ((peak - paths) / peak).max(axis=1)


def parametric_var(mu_p, sigma_p, alpha, h, v0):
    z = norm.ppf(alpha)
    return v0 * (z * sigma_p * np.sqrt(h) - mu_p * h)


def parametric_cvar(mu_p, sigma_p, alpha, h, v0):
    z = norm.ppf(alpha)
    return v0 * (sigma_p * np.sqrt(h) * norm.pdf(z) / (1 - alpha) - mu_p * h)


def var_ci(pnl, alpha, n_boot=200, seed=0):
    """95% bootstrap confidence interval on the VaR estimate."""
    rng = np.random.default_rng(seed)
    stats = [var(rng.choice(pnl, size=len(pnl)), alpha) for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])