import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)


def plot_pnl(pnl, v, c, alpha, path="outputs/pnl_dist.png"):
    plt.figure(figsize=(9, 5))
    plt.hist(pnl, bins=200, color="steelblue", alpha=0.75)
    plt.axvline(-v, color="red", ls="--", label=f"VaR {alpha:.0%}: {v:,.0f}")
    plt.axvline(-c, color="black", ls="--", label=f"CVaR {alpha:.0%}: {c:,.0f}")
    plt.title("Simulated horizon P&L distribution")
    plt.xlabel("P&L"); plt.ylabel("Frequency"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_paths(paths, path="outputs/paths.png", n_show=200):
    plt.figure(figsize=(9, 5))
    plt.plot(paths[:n_show].T, color="grey", alpha=0.2, lw=0.7)
    q = np.percentile(paths, [5, 50, 95], axis=0)
    for line, lab, col in zip(q, ["5th", "Median", "95th"], ["red", "black", "green"]):
        plt.plot(line, color=col, lw=2, label=lab)
    plt.title("Simulated portfolio value paths")
    plt.xlabel("Day"); plt.ylabel("Value"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_drawdown(mdd, path="outputs/drawdown.png"):
    plt.figure(figsize=(9, 5))
    plt.hist(mdd * 100, bins=100, color="darkorange", alpha=0.8)
    plt.axvline(np.percentile(mdd, 95) * 100, color="red", ls="--", label="95th pct")
    plt.title("Maximum drawdown distribution")
    plt.xlabel("Max drawdown (%)"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_backtest(bt, alpha, path="outputs/backtest.png"):
    plt.figure(figsize=(11, 5))
    plt.plot(bt.index, bt["ret"] * 100, color="steelblue", lw=0.8, label="Daily return")
    plt.plot(bt.index, -bt["var"] * 100, color="red", lw=1, label=f"-VaR {alpha:.0%}")
    b = bt[bt["breach"]]
    plt.scatter(b.index, b["ret"] * 100, color="black", s=20, zorder=3, label="Breach")
    plt.title("1-day VaR backtest"); plt.ylabel("%"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()