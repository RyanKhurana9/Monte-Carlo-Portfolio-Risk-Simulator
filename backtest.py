import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import xlogy
from simulate import simulate_asset_returns


def kupiec_pof(x, n, p):
    """Kupiec proportion-of-failures likelihood-ratio test."""
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - chi2.cdf(lr, df=1)


def rolling_backtest(log_rets: pd.DataFrame, weights, window=250, alpha=0.99,
                     n_sims=10_000, method="normal"):
    arr = log_rets.values
    actual = np.expm1(arr) @ weights
    rows = []
    for t in range(window, len(arr)):
        hist = arr[t - window:t]
        mu, cov = hist.mean(0), np.cov(hist.T)
        sims = simulate_asset_returns(mu, cov, n_sims, 1, method, hist=hist)
        pnl = np.expm1(sims[:, 0, :]) @ weights
        v = -np.quantile(pnl, 1 - alpha)
        rows.append((log_rets.index[t], actual[t], v))
    out = pd.DataFrame(rows, columns=["date", "ret", "var"]).set_index("date")
    out["breach"] = out["ret"] < -out["var"]
    return out