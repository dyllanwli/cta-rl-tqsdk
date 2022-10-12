from typing import Dict, List
import logging
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
from .constant import EnvConfig

from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account, Quote
from tqsdk.tafunc import time_to_datetime, time_to_s_timestamp
from commodity import Commodity
from .constant import EnvConfig

from dao.mongo import MongoDAO
from utils import Interval


def get_symbols_by_names(config: EnvConfig):
    # get instrument symbols
    cmod = Commodity()
    return [cmod.get_instrument_name(name) for name in config.symbols]


duration_seconds: Dict[str, int] = {
    Interval.ONE_SEC: 1,
    Interval.FIVE_SEC: 5,
    Interval.ONE_MIN: 60,
    Interval.FIVE_MIN: 300,
    Interval.FIFTEEN_MIN: 900,
    Interval.THIRTY_MIN: 1800,
    Interval.ONE_HOUR: 3600,
    Interval.FOUR_HOUR: 14400,
    Interval.ONE_DAY: 86400,
}

class TargetPosTaskOffline:
    def __init__(self, commission: float = 5.0, verbose: int = 1):
        self.last_volume = 0
        self.positions = deque([])
        self.commission = commission
        self.margin_rate = 2.0
        if verbose == 0:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    
    def insert_order(self,):
        # TODO insert order
        pass

    def set_target_volume(self, volume, price: float):
        profit = 0
        if self.last_volume == volume:
            logging.debug("hold position")
            return 0
        if volume * self.last_volume > 0:
            if volume > 0:
                position_change = volume - self.last_volume
                if position_change > 0:
                    logging.debug("bug long %d", position_change)
                    for _ in range(position_change):
                        self.positions.append(price)
                else:
                    logging.debug("sell long %d", position_change)
                    for _ in range(-position_change):
                        profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
            else:
                position_change = self.last_volume - volume
                if position_change > 0:
                    logging.debug("buy short %d", position_change)
                    for _ in range(position_change):
                        self.positions.append(price)
                else:
                    logging.debug("sell short %d", position_change)
                    for _ in range(-position_change):
                        profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
        else:
            if volume >= 0:
                logging.debug("sell short %d", self.last_volume)
                while len(self.positions) > 0:
                    profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
                logging.debug("buy long %d", volume)
                for _ in range(volume):
                    self.positions.append(price)
            else:
                logging.debug("sell long %d", self.last_volume)
                while len(self.positions) > 0:
                    profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
                logging.debug("buy short %d", volume)
                for _ in range(-volume):
                    self.positions.append(price)
        self.last_volume = volume
        return profit

class DataLoader:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.is_random_sample = config.is_random_sample

        self.start_dt = self.config.start_dt
        self.end_dt = self.config.end_dt
        self.mongo = MongoDAO()

    def get_api(self) -> TqApi:
        return self._set_account(
            self.config.auth, self.config.backtest, self.config.init_balance)

    def get_offline_data(self, interval: Interval, instrument_id: str, offset_bar_length: int) -> pd.DataFrame:
        if self.is_random_sample:
            offset = self.config.max_steps + offset_bar_length + 100
            start_dt = datetime.combine(self.start_dt, datetime.min.time())
            end_dt = datetime.combine(self.end_dt, datetime.max.time())

            low_dt = time_to_s_timestamp(start_dt)
            high_dt = time_to_s_timestamp(end_dt) - offset

            sample_start = np.random.randint(low = low_dt, high = high_dt, size=1)[0]
            sample_start_dt = time_to_datetime(sample_start).date()
            print("dataloader: Loading random offline data from ", sample_start_dt)
            df = self.mongo.load_bar_data(
                instrument_id, sample_start_dt, self.end_dt, interval, limit=offset)
            print("dataloader: random offline data loaded, shape: ", df.shape)
            return df
        else:
            print("dataloader: Loading offline data...")
            df = self.mongo.load_bar_data(
                instrument_id, self.start_dt, self.end_dt, interval)
            return df

    def _set_account(self, auth, backtest, init_balance, live_market=None, live_account=None):
        """
        Set account and API for TqApi
        If you want to use free backtest, remove backtest parameter
        e.g. 
            api: TqApi = TqApi(auth=auth, account=TqSim(init_balance=init_balance))
        """
        api = None
        if backtest is not None:
            # backtest
            print("Backtest mode")
            api: TqApi = TqApi(auth=auth, account=TqSim(init_balance=init_balance), 
                backtest=backtest,
            )
        else:
            # live or sim
            if live_market:
                print("Live market mode")
                api = TqApi(
                    account=live_account, auth=auth)
            else:
                print("Sim mode")
                api = TqApi(auth=auth, account=TqSim(
                    init_balance=init_balance))
        return api