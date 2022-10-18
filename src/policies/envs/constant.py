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
                 max_steps: int = 720,  # max actions per episode
                 is_offline: bool = False,
                 is_random_sample: bool = False,
                 project_name: str = "futures-trading-4",
                 ):
        self.auth: TqAuth = auth
        self.symbols: List[str] = symbols
        self.start_dt: date = start_dt
        self.end_dt: date = end_dt
        self.live_market: bool = live_market
        self.live_account: TqAccount = live_account
        self.wandb = wandb
        self.project_name = project_name

        # dataloader settings
        self.max_steps: int = max_steps
        self.is_offline: bool = is_offline
        self.is_random_sample: bool = is_random_sample

        # other config params
        self.init_balance: float = 2000000
        self.max_volume: int = 1
        self.trade_position_ratio_limit: float = 0.9

        INTERVAL = Interval()

        self.data_length: Dict[str, int] = {
            INTERVAL.ONE_SEC : 1,
            INTERVAL.FIVE_SEC : 1,
            INTERVAL.ONE_MIN : 1,
            INTERVAL.FIVE_MIN : 1,
            INTERVAL.FIFTEEN_MIN : 1,
            INTERVAL.THIRTY_MIN : 1,
            INTERVAL.ONE_HOUR : 1,
            INTERVAL.FOUR_HOUR : 1,
            INTERVAL.ONE_DAY : 1,
        }

        self.action_space_type = "discrete" # "discrete" or "continuous"

    @property
    def backtest(self) -> TqBacktest:
        if self.start_dt is None or self.end_dt is None:
            return None
        else:
            return TqBacktest(self.start_dt, self.end_dt)
