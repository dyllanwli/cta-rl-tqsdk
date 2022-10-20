from typing import Dict, List, NamedTuple
import logging
from datetime import datetime, timedelta
import time
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

def get_symbols_by_names(config: EnvConfig):
    # get instrument symbols
    cmod = Commodity()
    return [cmod.get_instrument_name(name) for name in config.symbols]

class TargetPosTaskOffline:
    def __init__(self, commission: float = 5.0, verbose: int = 1):
        self.last_volume = 0
        self.positions = deque([])
        self.commission = commission
        self.margin_rate = 4.0
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
        if volume > 0 and self.last_volume >= 0:
            position_change = volume - self.last_volume
            if position_change > 0:
                logging.debug("buy long %d", position_change)
                for _ in range(position_change):
                    self.positions.append(price)
            else:
                logging.debug("sell long %d", position_change)
                for _ in range(-position_change):
                    profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
        elif volume < 0 and self.last_volume <= 0:
            position_change = self.last_volume - volume
            if position_change > 0:
                logging.debug("buy short %d", position_change)
                for _ in range(position_change):
                    self.positions.append(price)
            else:
                logging.debug("sell short %d", position_change)
                for _ in range(-position_change):
                    profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
        elif volume >= 0 and self.last_volume < 0:
            logging.debug("sell all short %d", self.last_volume)
            while len(self.positions) > 0:
                profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
            logging.debug("buy long %d", volume)
            for _ in range(volume):
                self.positions.append(price)
        elif volume <= 0 and self.last_volume > 0:
            logging.debug("sell all long %d", self.last_volume)
            while len(self.positions) > 0:
                profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
            logging.debug("buy short %d", volume)
            for _ in range(-volume):
                self.positions.append(price)
        self.last_volume = volume
        return profit

class SimpleTargetPosTaskOffline:
    """
    Simple target position task only deal with three actions:
    1. hold position
    2. long position
    3. short position
    """
    def __init__(self, commission: float = 5.0, verbose: int = 1):
        self.positions = deque([])
        self.commission = commission
        self.margin_rate = 4.0
        self.last_action = 0
        if verbose == 0:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    
    def insert_order(self,):
        # TODO insert order
        pass

    def set_target_volume(self, volume, price: float):
        profit = 0
        if volume == 0:
            logging.debug("hold position")
        elif volume > 0:
            logging.debug("buy long %d", volume)
            if self.last_action < 0:
                # if last volume is short, sell all short positions
                profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
            else:
                self.positions.append(price)
        else:
            logging.debug("buy short %d", -volume)
            if self.last_action > 0:
                # if last volume is long, sell all long positions
                profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
            else:
                # buy short positions
                self.positions.append(price)
        self.last_action = volume
        return profit

class DataLoader:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.is_random_sample = config.is_random_sample

        self.start_dt = datetime.combine(self.config.start_dt, datetime.min.time())
        self.end_dt = datetime.combine(self.config.end_dt , datetime.max.time())
        self.mongo = MongoDAO()

    def get_api(self) -> TqApi:
        return self._set_account(
            self.config.auth, self.config.backtest, self.config.init_balance)

    def get_offline_data(self, interval: str, instrument_id: str, offset: int) -> pd.DataFrame:
        if self.is_random_sample:
            while True:
                low_dt = time_to_s_timestamp(self.start_dt)
                high_dt = time_to_s_timestamp(self.end_dt) - offset - 300

                sample_start = np.random.randint(low = low_dt, high = high_dt, size=1)[0]
                start_dt: datetime = time_to_datetime(sample_start)
                # start_dt = start_dt + timedelta(hours=7)
                # print("dataloader: Loading random offline data from ", sample_start_dt)
                try:
                    df = self.mongo.load_bar_data(
                        instrument_id, start_dt, self.end_dt, interval, limit=offset)
                    assert df.shape[0] >= offset
                    break
                except Exception as e:
                    print("dataloader: random offline data load failed, retrying...")

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