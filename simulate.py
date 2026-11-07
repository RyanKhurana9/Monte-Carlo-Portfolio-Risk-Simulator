import numpy as np


def simulate_asset_returns(mu, cov, n_sims, horizon, method="normal",
                           df=5, hist=None, seed=None):
    """Returns array (n_sims, horizon, n_assets) of daily LOG returns."""
    rng = np.random.default_rng(seed)
    k = len(mu)

    if method == "bootstrap":
        idx = rng.integers(0, len(hist), size=(n_sims, horizon))
        return hist[idx]                                   # historical simulation

    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_sims, horizon, k))
    corr_z = z @ L.T                                       # correlated N(0, Sigma)

    if method == "normal":
        return mu + corr_z

    if method == "t":                                      # multivariate Student-t
        w = rng.chisquare(df, size=(n_sims, horizon, 1))
        scale = np.sqrt((df - 2) / df)                     # keeps Cov = Sigma
        return mu + corr_z * scale / np.sqrt(w / df)

    raise ValueError(method)


def portfolio_paths(asset_log_ret, weights, v0):
    """Daily-rebalanced portfolio value paths, shape (n_sims, horizon+1)."""
    simple = np.expm1(asset_log_ret)                       # log -> simple returns
    rp = simple @ weights                                  # portfolio simple return
    vals = v0 * np.cumprod(1.0 + rp, axis=1)
    start = np.full((vals.shape[0], 1), v0)
    return np.concatenate([start, vals], axis=1)