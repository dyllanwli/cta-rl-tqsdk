from typing import List, Dict
from datetime import date
from ray.rllib.env.env_context import EnvContext
from tqsdk import TqAuth, TqApi, 

class EnvConfig(EnvContext):
    # env config is a entity
    def __init__(self, 
        api: TqAuth,
        symbols: List[str],
        max_steps: int = 30000, # max actions per episode
    ):
        self.api: TqApi = api
        self.symbols: List[str] = symbols
        self.max_steps: int = max_steps

        # other config params
        self.init_balance: float = 200000
        self.max_volume: int = 30
        self.trade_position_ratio_limit: float = 0.9

        self.data_length: Dict[str, int] = {
            "ticks": 30,
            "bar_1m": 30,
            "bar_5m": 30,
            "bar_30m": 30,
            "bar_60m": 30,
            "bar_1d": 30,
        }






