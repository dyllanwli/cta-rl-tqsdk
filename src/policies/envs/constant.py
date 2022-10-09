from tkinter import Variable
from typing import List, Dict
from datetime import date
from enum import Enum
from tqsdk import TqAuth, TqSim, TqBacktest, BacktestFinished, TqAccount, TqApi
from utils import Interval


class EnvConfig:
    # env config is a entity
    def __init__(self,
                 auth: TqAuth,
                 symbols: List[str],
                 start_dt: date = None,
                 end_dt: date = None,
                 live_market: bool = False,
                 live_account: TqAccount = None,
                 max_steps: int = 30000,  # max actions per episode
                 dataloader: str = "db",
                 intervals: List[Interval] = [Interval.ONE_MIN, Interval.THIRTY_MIN],
                 ):
        self.auth: TqAuth = auth
        self.symbols: List[str] = symbols
        self.start_dt: date = start_dt
        self.end_dt: date = end_dt
        self.live_market: bool = live_market
        self.live_account: TqAccount = live_account
        self.max_steps: int = max_steps

        self.dataloader: str = dataloader  # db or api
        self.intervals: List[Interval] = intervals

        # other config params
        self.init_balance: float = 2000000
        self.max_volume: int = 30
        self.trade_position_ratio_limit: float = 0.9

        self.wandb_log: bool = True

        self.data_length: Dict[str, int] = {
            Interval.ONE_SEC: 10,
            Interval.FIVE_SEC: 10,
            Interval.ONE_MIN: 10,
            Interval.FIVE_MIN: 10,
            Interval.FIFTEEN_MIN: 10,
            Interval.THIRTY_MIN: 10,
            Interval.ONE_HOUR: 10,
            Interval.FOUR_HOUR: 10,
            Interval.ONE_DAY: 10,
        }

    @property
    def backtest(self) -> TqBacktest:
        if self.start_dt is None or self.end_dt is None:
            return None
        else:
            return TqBacktest(self.start_dt, self.end_dt)
