from tqsdk.ta import *
import pandas as pd


class Factors:
    def __init__(self):
        pass
    
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
        boll_bottom = boll["bottom"].iloc[-1]
        return [boll_top, boll_bottom]
    
    def bias(self, df: pd.DataFrame, n: int = 6):
        bias = BIAS(df, n)["bias"].iloc[-1]
        return [bias]