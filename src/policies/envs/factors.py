from tqsdk.ta import MACD, RSI
import pandas as pd


class Factors:
    def __init__(self):
        pass
    
    def macd_bar(self, df: pd.DataFrame, short: int = 30, long: int = 60, m: int = 20):
        macd = MACD(df, short, long, m)
        return list(macd["bar"])

    def rsi(self, df: pd.DataFrame, n: int = 20):
        print(df)
        df.to_csv('test.csv')
        rsi = RSI(df, n)
        return list(rsi["rsi"])