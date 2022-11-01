from datetime import date, datetime

import pandas as pd
import numpy as np
from policies.envs.tools import DataLoader, get_symbols_by_names
from policies.envs.constant import EnvConfig
from utils.utils import Interval, max_step_by_day
from utils.api import API
from .algos import Algos


class ModelTrainer:
    def __init__(self, account = "a1", train_type = "tune", max_sample_size = 1e6):
        print("Initializing Model trainer")
        auth = API(account=account).auth
        self.train_type = train_type  # tune or train
        self.algo_name = "TFT"
        self.algo = Algos(self.algo_name)
        
        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") if self.train_type == "train" else False
        self.project_name = "futures-alpha-8"
        INTERVAL = Interval()
        self.interval = INTERVAL.ONE_MIN
        self.max_steps = max_step_by_day[self.interval]
        self.training_iteration = dict({
            INTERVAL.ONE_MIN: 100,
            INTERVAL.FIVE_SEC: 400,
            INTERVAL.ONE_SEC: 500,
        })
        self.symbol = get_symbols_by_names(["cotton"])[0]
        self.max_sample_size = int(max_sample_size)
    
    def get_training_data(self, start_dt=date(2016, 1, 1), end_dt=date(2022, 7, 1)):
        dataloader = DataLoader(start_dt=start_dt, end_dt=end_dt)
        offline_data: pd.DataFrame = dataloader.get_offline_data(
                interval=self.interval, instrument_id=self.symbol, offset=self.max_sample_size, fixed_dt=True)
        return offline_data
    
    def run(self):
        data = self.get_training_data()
        model = self.algo.get_model(data)
        model.print_baseline()
        # model.get_optimal_lr()
        trainer = model.train()
        # model.test_predict(checkpoint_path="/h/diya.li/quant/tqsdk-rl/src/lightning_logs/lightning_logs/version_0/checkpoints/epoch=26-step=2700.ckpt")
        model.test_predict(trainer)
        print("Done")
        model.tune()

        # predict with new data
        # new_data = self.get_training_data(start_dt=date(2022, 7, 1), end_dt=date(2022, 8, 1))