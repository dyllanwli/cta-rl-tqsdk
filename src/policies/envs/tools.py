from typing import Dict, List
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
from .constant import EnvConfig

from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account, Quote
from tqsdk.tafunc import time_to_datetime
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
    # TODO: add offline datasource on env v3
    def __init__(self, config: EnvConfig):
        self.dataloader = config.dataloader

        self._prepare_data(config)

        self.max_retry = 3

    def _prepare_data(self, config: EnvConfig):
        print("Preparing data...")
        self.intervals = config.intervals
        self.data_length = config.data_length
        self.start_dt = config.start_dt
        self.end_dt = config.end_dt
        self.OHLCV = ['open', 'high', 'low', 'close', 'volume']
        self.instrument_list = get_symbols_by_names(config)
        if self.dataloader == "api":
            self.api = self._set_account(
                config.auth, config.backtest, config.init_balance, config.live_market, config.live_account)
            
            self.instrument_quotes: Dict[str, Quote] = dict()
            self.data: Dict[str, Dict[str, pd.DataFrame]] = dict()
            for instrument_id in self.instrument_list:
                self.instrument_quotes[instrument_id] = self.api.get_quote(
                    instrument_id)
                self._update_subscription(
                    instrument_id, self.instrument_quotes, self.intervals)

        elif self.dataloader == "db":
            self.mongo = MongoDAO()
            self.data: Dict[str, Dict[str, pd.DataFrame]] = dict()
            for instrument_id in self.instrument_list:
                self.data[instrument_id] = dict()
                for interval in self.intervals:
                    df = self.mongo.load_bar_data(
                        instrument_id, self.start_dt, self.end_dt, interval)
                    self.data[instrument_id][interval] = df[self.OHLCV].to_numpy(dtype=np.float64)



    def _set_account(self, auth, backtest, init_balance, live_market=None, live_account=None):
        """
        Set account and API for TqApi
        """
        api = None
        if backtest is not None:
            # backtest
            print("Backtest mode")
            api: TqApi = TqApi(auth=auth, backtest=backtest,
                               account=TqSim(init_balance=init_balance))
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

    def _update_subscription(self, instrument_id: str, instrument_quotes: Dict[str, Quote], intervals: List[Interval]):
        # update subscription
        underlying_symbol = instrument_quotes[instrument_id].underlying_symbol
        for interval in intervals:
            self.data[instrument_id] = dict()
            self.data[instrument_id][interval] = self.api.get_kline_serial(
                underlying_symbol, duration_seconds[interval], self.data_length[interval])

    def wait_update(self):
        if self.dataloader == "api":
            self.api.wait_update()
            for instrument_id in self.instrument_list:
                if self.api.is_changing(self.instrument_quotes[instrument_id], "underlying_symbol"):
                    self._update_subscription(
                        instrument_id, self.instrument_quotes, self.intervals)
        elif self.dataloader == "db":
            pass

    def get_data(self):
        states: Dict[str, Dict[str, pd.DataFrame]] = dict()
        if self.dataloader == "api":
            self.api.wait_update()
            for instrument_id in self.instrument_list:
                for interval in self.intervals:
                    retry = 0
                    while retry < self.max_retry:
                        states[instrument_id] = dict()
                        quote = self.data[instrument_id][interval].iloc[-self.data_length[interval]:]
                        states[instrument_id][interval] = quote[self.OHLCV].to_numpy(dtype=np.float64)
                        if not np.isnan(states[instrument_id][interval]).any():
                            break
                        else:
                            print("Nan in state data, retrying...")
                            self.api.wait_update()
                            retry += 1
                            if retry == self.max_retry:
                                raise Exception("Nan in state data, retrying...")
            return states
        elif self.dataloader == "db":
            for instrument_id in self.instrument_list:
                states[instrument_id] = dict()
                for interval in self.intervals:
                    states[instrument_id][interval] = self.data[instrument_id][interval].iloc[-self.data_length[interval]:]
                    
            return states
            

    def get_account(self):
        if self.dataloader == "api":
            return self.api.get_account()
        elif self.dataloader == "db":
            return None
