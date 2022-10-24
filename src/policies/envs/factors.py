from tqsdk.ta import *
import pandas as pd
import numpy as np
from copy import deepcopy

class Factors:
    def __init__(self, obs_space, factor_length):
        self.factors_set = set(obs_space.keys())
        self.factor_length = factor_length
        self.r_cols = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
        self.n_cols = ["n_open", "n_high", "n_low", "n_close", "n_volume", "n_open_oi", "n_close_oi"]
        self.OHLCV = ['open', 'high', 'low', 'close', 'volume']
        self.n_OHLCV = ["n_open", "n_high", "n_low", "n_close", "n_volume"]
    
    def set_ohlcv_state(self, ohlcv: np.ndarray):
        state = {
            "open": ohlcv[:, 0],
            "high": ohlcv[:, 1],
            "low": ohlcv[:, 2],
            "close": ohlcv[:, 3],
            "volume": ohlcv[:, 4],
        }
        return state
    
    def set_state_factors(self, bar_data, last_price):
        """
        :param bar_data: bar data
        :param last_price: last price
        """
        info = dict()
        state = dict()
        if "bias" in self.factors_set:
            factor = np.array(self.bias(
                bar_data, n=7), dtype=np.float32)
            state["bias"] = factor
            info["factors/bias"] = factor[0]
        if "macd_bar" in self.factors_set:
            factor = np.array(self.macd_bar(
                bar_data, short=30, long=60, m=15), dtype=np.float32)[-self.factor_length:]
            state["macd_bar"] = factor
            info["factors/macd_bar"] = factor[-1]
        if "boll" in self.factors_set:
            factor = np.array(self.boll_residual(
                bar_data, n=26, p=5, price = last_price), dtype=np.float32)
            state["boll"] = factor
            info["factors/boll_mid"] = factor[1]
        if "kdj" in self.factors_set:
            factor = np.array(self.kdj(
                bar_data, n=9, m1=3, m2=3), dtype=np.float32)
            state["kdj"] = factor
            info["factors/kdj_k"] = factor[0]
        return state, info
    
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
    # ############################## Define Custom Factors ##############################
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
    
    def bias(self, df: pd.DataFrame, n: int = 7):
        bias = BIAS(df, n)["bias"].iloc[-1]
        return [bias]

    def kdj(self, df, n=9, m1=3, m2=3):
        kdj = KDJ(df, n, m1, m2)
        k = kdj["k"].iloc[-1]
        d = kdj["d"].iloc[-1]
        j = kdj["j"].iloc[-1]
        return [k, d, j]