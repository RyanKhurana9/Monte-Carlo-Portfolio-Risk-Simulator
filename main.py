import numpy as np
import pandas as pd
from data import load_prices, log_returns
from simulate import simulate_asset_returns, portfolio_paths
from risk import (var, cvar, max_drawdown, parametric_var,
                  parametric_cvar, var_ci)
from backtest import rolling_backtest, kupiec_pof
import plots

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "GLD"]
WEIGHTS = np.full(len(TICKERS), 1 / len(TICKERS))
START = "2018-01-01"
V0 = 1_000_000
HORIZON = 10          # trading days
N_SIMS = 100_000
SEED = 42


def main():
    prices = load_prices(TICKERS, START)
    lr = log_returns(prices)
    hist = lr.values
    mu, cov = hist.mean(0), np.cov(hist.T)

    # historical portfolio stats (for parametric benchmark)
    rp_hist = np.expm1(hist) @ WEIGHTS
    mu_p, sigma_p = rp_hist.mean(), rp_hist.std(ddof=1)

    rows = []
    store = {}
    for method in ["normal", "t", "bootstrap"]:
        sims = simulate_asset_returns(mu, cov, N_SIMS, HORIZON, method,
                                      df=5, hist=hist, seed=SEED)
        paths = portfolio_paths(sims, WEIGHTS, V0)
        pnl = paths[:, -1] - V0
        mdd = max_drawdown(paths)
        lo, hi = var_ci(pnl, 0.99)
        rows.append({
            "method": method,
            "VaR95": var(pnl, 0.95), "VaR99": var(pnl, 0.99),
            "VaR99_CI": f"[{lo:,.0f}, {hi:,.0f}]",
            "CVaR95": cvar(pnl, 0.95), "CVaR99": cvar(pnl, 0.99),
            "MDD_mean%": mdd.mean() * 100, "MDD_95th%": np.percentile(mdd, 95) * 100,
        })
        store[method] = (paths, pnl, mdd)

    rows.append({
        "method": "parametric",
        "VaR95": parametric_var(mu_p, sigma_p, 0.95, HORIZON, V0),
        "VaR99": parametric_var(mu_p, sigma_p, 0.99, HORIZON, V0),
        "VaR99_CI": "-",
        "CVaR95": parametric_cvar(mu_p, sigma_p, 0.95, HORIZON, V0),
        "CVaR99": parametric_cvar(mu_p, sigma_p, 0.99, HORIZON, V0),
        "MDD_mean%": np.nan, "MDD_95th%": np.nan,
    })

    pd.options.display.float_format = "{:,.1f}".format
    print(f"\n{HORIZON}-day risk on {V0:,} portfolio, {N_SIMS:,} sims\n")
    print(pd.DataFrame(rows).set_index("method").to_string())

    # plots (Student-t run as the headline fat-tailed model)
    paths, pnl, mdd = store["t"]
    plots.plot_pnl(pnl, var(pnl, 0.99), cvar(pnl, 0.99), 0.99)
    plots.plot_paths(paths)
    plots.plot_drawdown(mdd)

    # backtest: 1-day 99% VaR, rolling 250-day window
    print("\nBacktest (1-day, 99%, 250d window)")
    for method in ["normal", "t", "bootstrap"]:
        bt = rolling_backtest(lr, WEIGHTS, 250, 0.99, 10_000, method)
        x, n = int(bt["breach"].sum()), len(bt)
        stat, pval = kupiec_pof(x, n, 0.01)
        print(f"{method:10s} breaches={x}/{n} ({x/n:.2%}) "
              f"expected={0.01*n:.1f} Kupiec LR={stat:.2f} p={pval:.3f}")
        if method == "normal":
            plots.plot_backtest(bt, 0.99)


if __name__ == "__main__":
    main()