from datetime import date, datetime

import pandas as pd
import numpy as np
from policies.envs.tools import DataLoader, get_symbols_by_names
from policies.envs.constant import EnvConfig
from utils.utils import Interval, max_step_by_day
from utils.api import API


class ModelTrainer:
    def __init__(self, account = "a1", train_type = "tune"):
        print("Initializing Model trainer")
        auth = API(account=account).auth
        self.train_type = train_type  # tune or train
        self.algo_name = "TFT"

        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") if self.train_type == "train" else False
        self.project_name = "futures-alpha-8"
        INTERVAL = Interval()
        self.interval = INTERVAL.ONE_SEC
        self.max_steps = max_step_by_day[self.interval]
        self.training_iteration = dict({
            INTERVAL.ONE_MIN: 100,
            INTERVAL.FIVE_SEC: 400,
            INTERVAL.ONE_SEC: 500,
        })
        # only trainer mode will log to wandb in env
        self.config = EnvConfig({
            "auth": auth,
            "symbols": ["cotton"],
            # "symbols": ["sliver"],
            "start_dt": date(2016, 1, 1),
            "end_dt": date(2022, 8, 1),
            "live_market": False,
            "live_account": None,
            "wandb": self.wandb_name,
            "is_offline": True,
            "max_sample_size": 1e6,
            "project_name": self.project_name,
            "interval": self.interval,
            "max_steps": self.max_steps,
            "high_freq": True,
        })
        self.dataloader = DataLoader(self.config)
        self.interval: str = self.config.interval # interval for OHLCV
        self.symbol = get_symbols_by_names(self.config)[0]
        self.max_sample_size = self.config.max_sample_size
    
    def get_training_data(self):
        offline_data: pd.DataFrame = self.dataloader.get_offline_data(
                interval=self.interval, instrument_id=self.symbol, offset=self.max_sample_size)
        return offline_data