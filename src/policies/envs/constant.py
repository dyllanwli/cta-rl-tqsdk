from typing import List, Dict
from datetime import date
from tqsdk import TqAuth, TqSim, TqBacktest, BacktestFinished, TqAccount, TqApi


class EnvConfig:
    # env config is a entity
    def __init__(self,
                 auth: TqAuth,
                 symbols: List[str],
                 backtest: TqBacktest = None,
                 live_market: bool = False,
                 live_account: TqAccount = None,
                 max_steps: int = 30000,  # max actions per episode
                 ):
        self.auth: TqAuth = auth
        self.symbols: List[str] = symbols
        self.backtest: TqBacktest = backtest
        self.live_market: bool = live_market
        self.live_account: TqAccount = live_account
        self.max_steps: int = max_steps

        # other config params
        self.init_balance: float = 2000000
        self.max_volume: int = 30
        self.trade_position_ratio_limit: float = 0.9

        self.wandb_log: bool = True

        self.data_length: Dict[str, int] = {
            "ticks": 2,
            "bar_1s": 2,
            "bar_1m": 2,
            "bar_5m": 2,
            "bar_30m": 2,
            "bar_60m": 2,
            "bar_1d": 2,
        }
