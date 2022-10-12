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
                 wandb = None,
                 max_steps: int = 12000,  # max actions per episode
                 is_offline: bool = False,
                 is_random_sample: bool = False,
                 ):
        self.auth: TqAuth = auth
        self.symbols: List[str] = symbols
        self.start_dt: date = start_dt
        self.end_dt: date = end_dt
        self.live_market: bool = live_market
        self.live_account: TqAccount = live_account
        self.wandb = wandb

        # dataloader settings
        self.max_steps: int = max_steps
        self.is_offline: bool = is_offline
        self.is_random_sample: bool = is_random_sample

        # other config params
        self.init_balance: float = 2000000
        self.max_volume: int = 10
        self.trade_position_ratio_limit: float = 0.9

        self.data_length: Dict[str, int] = {
            Interval.ONE_SEC.value: 5,
            Interval.FIVE_SEC.value: 5,
            Interval.ONE_MIN.value: 5,
            Interval.FIVE_MIN.value: 5,
            Interval.FIFTEEN_MIN.value: 5,
            Interval.THIRTY_MIN.value: 5,
            Interval.ONE_HOUR.value: 5,
            Interval.FOUR_HOUR.value: 5,
            Interval.ONE_DAY.value: 5,
        }

        self.action_space_type = "discrete" # "discrete" or "continuous"

    @property
    def backtest(self) -> TqBacktest:
        if self.start_dt is None or self.end_dt is None:
            return None
        else:
            return TqBacktest(self.start_dt, self.end_dt)
