from typing import List, Dict, NamedTuple
from datetime import date
from enum import Enum
from tqsdk import TqAuth, TqSim, TqBacktest, BacktestFinished, TqAccount, TqApi
from utils import Interval, InitOverallStep

class MaxStepByDay(NamedTuple):
    ONE_SEC: int = 20700
    FIVE_SEC: int = 4140
    ONE_MIN: int = 345
    FIVE_MIN: int = 72
    FIFTEEN_MIN: int = 24

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

        # rl settings
        self.action_space_type = "discrete" # "discrete" or "continuous"

        # other config params
        self.init_balance: float = 2000000
        self.max_volume: int = 1
        self.trade_position_ratio_limit: float = 0.9

        INTERVAL = Interval()
        INIT_STEP = InitOverallStep()
        MAXSTEP = MaxStepByDay()

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
        # subscribed interval
        self.interval_1: str = INTERVAL.ONE_MIN
        self.init_step_1: int = 120 # 2 hours

        # dataloader settings
        self.max_steps: int = MAXSTEP.ONE_MIN * 3 # max actions per episode
        self.is_offline: bool = is_offline
        self.is_random_sample: bool = is_random_sample
        

    @property
    def backtest(self) -> TqBacktest:
        if self.start_dt is None or self.end_dt is None:
            return None
        else:
            return TqBacktest(self.start_dt, self.end_dt)

