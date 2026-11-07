import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(tickers, start, end=None):
    px = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    return px.reindex(columns=tickers).dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()