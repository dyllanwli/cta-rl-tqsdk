from tqsdk.ta import *
import pandas as pd


class Factors:
    def __init__(self):
        pass
    
    def normalize(self, x):
        if isinstance(x, np.ndarray):
            min_val = np.min(x)
            max_val = np.max(x)
            return (x - min_val) / (max_val - min_val)
        elif isinstance(x, pd.DataFrame):
            d = x.copy()
            cols = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
            for col in cols:
                d[col] = (d[col] - x[col].min()) / (x[col].max() - x[col].min())
            return d


    def macd_bar(self, df: pd.DataFrame, short: int = 30, long: int = 60, m: int = 20):
        macd = list(MACD(df, short, long, m)["bar"])
        return macd

    def rsi(self, df: pd.DataFrame, n: int = 7):
        rsi = RSI(df, n)["rsi"].iloc[-1]
        # to deal with shift
        return [0] if np.isnan(rsi) else [rsi]
    
    def boll(self, df: pd.DataFrame, n: int = 26, p: int = 5):
        boll = BOLL(df, n, p)
        boll_top = boll["top"].iloc[-1]
        boll_mid = boll["mid"].iloc[-1]
        boll_bottom = boll["bottom"].iloc[-1]
        return [
            boll_top if not np.isnan(boll_top) else 0, 
            boll_mid if not np.isnan(boll_mid) else 0,
            boll_bottom if not np.isnan(boll_bottom) else 0
        ]
    
    def bias(self, df: pd.DataFrame, n: int = 6):
        bias = BIAS(df, n)["bias"].iloc[-1]
        return [bias]

    def kdj(self, df, n=9, m1=3, m2=3):
        kdj = KDJ(df, n, m1, m2)
        k = kdj["k"].iloc[-1]
        d = kdj["d"].iloc[-1]
        j = kdj["j"].iloc[-1]
        return [k, d, j]