from tqsdk.ta import *
import pandas as pd
from copy import deepcopy

class Factors:
    def __init__(self):
        self.r_cols = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
        self.n_cols = ["n_open", "n_high", "n_low", "n_close", "n_volume", "n_open_oi", "n_close_oi"]
    
    def mean_normalize(self, x):
        if isinstance(x, np.ndarray):
            mean = np.mean(x, axis=0)
            normalized = x / mean
            return np.nan_to_num(normalized, nan=0)
        elif isinstance(x, pd.DataFrame):
            d = deepcopy(x)
            d[self.n_cols] = d[self.r_cols].apply(lambda y: y - np.mean(y), axis=0)
            return d.fillna(0)
    
    def mean_residual_normalize(self, x):
        if isinstance(x, np.ndarray):
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            normalized = (x - mean) / std
            return np.nan_to_num(normalized, nan=0)
        elif isinstance(x, pd.DataFrame):
            d = deepcopy(x)
            d[self.n_cols] = d[self.r_cols].apply(lambda y: (y - np.mean(y)) / np.std(y), axis=0)
            return d.fillna(0)

    def min_max_normalize(self, x):
        # normalize data by column
        if isinstance(x, np.ndarray):
            min_val = np.min(x, axis=0)
            max_val = np.max(x, axis=0)
            normalized = (x - min_val) / (max_val - min_val)
            return np.nan_to_num(normalized, nan=0)
        elif isinstance(x, pd.DataFrame):
            d = deepcopy(x)
            d[self.n_cols] = d[self.r_cols].apply(lambda y: (y - np.min(y)) / (np.max(y) - np.min(y)), axis=0)
            return d.fillna(0)

    def macd_bar(self, df: pd.DataFrame, short: int = 30, long: int = 60, m: int = 20):
        macd = list(MACD(df, short, long, m)["bar"])
        return macd

    def rsi(self, df: pd.DataFrame, n: int = 7):
        rsi = RSI(df, n)["rsi"].iloc[-1]
        # to deal with shift
        return [0] if np.isnan(rsi) else [rsi]
    
    def boll_residual(self, df: pd.DataFrame, n: int = 26, p: int = 5, price: float = 0.0):
        boll = BOLL(df, n, p)
        boll_top = boll["top"].iloc[-1]
        boll_mid = boll["mid"].iloc[-1]
        boll_bottom = boll["bottom"].iloc[-1]
        return [boll_top - price, boll_mid - price, boll_bottom - price]
    
    def bias(self, df: pd.DataFrame, n: int = 6):
        bias = BIAS(df, n)["bias"].iloc[-1]
        return [bias]

    def kdj(self, df, n=9, m1=3, m2=3):
        kdj = KDJ(df, n, m1, m2)
        k = kdj["k"].iloc[-1]
        d = kdj["d"].iloc[-1]
        j = kdj["j"].iloc[-1]
        return [k, d, j]