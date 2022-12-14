from typing import List, Dict
from datetime import date
from enum import Enum
from tqsdk import TqAuth, TqSim, TqBacktest, BacktestFinished, TqAccount, TqApi
from utils.utils import Interval, InitOverallStep


class EnvConfig:
    # env config is a entity
    def __init__(self, config):
        self.auth: TqAuth = config["auth"]
        self.symbols: List[str] = config["symbols"]
        self.start_dt: date = config["start_dt"]
        self.end_dt: date = config["end_dt"]
        self.live_market: bool = config["live_market"]
        self.live_account: TqAccount = config["live_account"]
        self.wandb = config["wandb"]
        self.project_name = config["project_name"]

        # rl settings
        self.action_space_type = "discrete" # "discrete" or "continuous"

        # other config params
        self.init_balance: float = 2000000
        self.max_volume: int = 1
        self.trade_position_ratio_limit: float = 0.9

        INTERVAL = Interval()
        # INIT_STEP = InitOverallStep()

        self.data_length: Dict[str, int] = {
            INTERVAL.ONE_SEC : 60,
            INTERVAL.FIVE_SEC : 60,
            INTERVAL.ONE_MIN : 60,
            INTERVAL.FIVE_MIN : 60,
            INTERVAL.FIFTEEN_MIN : 60,
            INTERVAL.THIRTY_MIN : 60,
            INTERVAL.ONE_HOUR : 60,
            INTERVAL.FOUR_HOUR : 60,
            INTERVAL.ONE_DAY : 60,
        }
        # subscribed interval
        self.interval: str = config["interval"]

        self.high_freq: bool = config["high_freq"]
        self.max_hold_steps: int = 30
        self.max_steps: int = config["max_steps"]
        self.is_offline: bool = config["is_offline"]
        self.max_sample_size: int = int(config["max_sample_size"])
        # set max sample size to reduce sample frequency
        

    @property
    def backtest(self) -> TqBacktest:
        if self.start_dt is None or self.end_dt is None:
            return None
        else:
            return TqBacktest(self.start_dt, self.end_dt)

