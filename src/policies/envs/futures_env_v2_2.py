from datetime import datetime
from collections import defaultdict, deque
import wandb
import numpy as np
from copy import deepcopy
import logging

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .constant import EnvConfig
from .tools import get_symbols_by_names

from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account, Quote
from tqsdk.tafunc import time_to_datetime, time_to_s_timestamp

from utils import Interval



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
                        self.insert_order() 
                        self.positions.append(price)
                else:
                    logging.debug("sell long %d", position_change)
                    for _ in range(-position_change):
                        self.insert_order()
                        profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
            else:
                position_change = self.last_volume - volume
                if position_change > 0:
                    logging.debug("buy short %d", position_change)
                    for _ in range(position_change):
                        self.insert_order()
                        self.positions.append(price)
                else:
                    logging.debug("sell short %d", position_change)
                    for _ in range(-position_change):
                        self.insert_order()
                        profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
        else:
            if volume >= 0:
                logging.debug("sell short %d", self.last_volume)
                while len(self.positions) > 0:
                    self.insert_order()
                    profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
                logging.debug("buy long %d", volume)
                for _ in range(volume):
                    self.insert_order()
                    self.positions.append(price)
            else:
                logging.debug("sell long %d", self.last_volume)
                while len(self.positions) > 0:
                    self.insert_order()
                    profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
                logging.debug("buy short %d", volume)
                for _ in range(-volume):
                    self.insert_order()
                    self.positions.append(price)
        self.last_volume = volume
        return profit


class FuturesEnvV2_2(gym.Env):
    """
    Custom Environment for RL training
    TqApi is required.
    Single symbol and interday only. 

    使用Offline TargetPosTask
    Reward设置为最后一次操作的收益
    """

    def __init__(self, config):
        super(gym.Env, self).__init__()
        config: EnvConfig = config['cfg']
        
        # wandb.init(project="futures-trading", name = config.wandb_name)

        self._skip_env_checking = True
        self._set_config(config)
        self.seed(42)

        self.reset()

    def _set_config(self, config: EnvConfig):
        # Subscribe instrument quote
        print("Setting config")
        self.api = self._set_account(
            config.auth, config.backtest, config.init_balance)
        self.account = self.api.get_account()

        # Set target position task to None to call _update_subscription at the first time
        self.target_pos_task:  TargetPosTaskOffline = None
        symbol = get_symbols_by_names(config)[0]
        self.instrument_quote = self.api.get_quote(symbol)
        self.OHLCV = ['open', 'high', 'low', 'close', 'volume']

        # Account and API
        self.data_length = config.data_length
        self.underlying_symbol = self.instrument_quote.underlying_symbol
        self.balance = deepcopy(self.account.balance)

        # RL config
        self.max_steps = config.max_steps
        self.action_space: spaces.Box = spaces.Box(
            low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int64)
        self.observation_space: spaces.Dict = spaces.Dict({
            "last_volume": spaces.Box(low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int64),
            "hour": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int64),
            "minute": spaces.Box(low=0, high=59, shape=(1,), dtype=np.int64),
            Interval.ONE_SEC.value: spaces.Box(low=-1, high=np.inf, shape=(self.data_length[Interval.ONE_SEC.value], 5), dtype=np.float64),
            Interval.ONE_MIN.value: spaces.Box(low=-1, high=np.inf, shape=(self.data_length[Interval.ONE_MIN.value], 5), dtype=np.float64),
            Interval.THIRTY_MIN.value: spaces.Box(low=-1, high=np.inf, shape=(self.data_length[Interval.THIRTY_MIN.value], 5), dtype=np.float64),
            # "bar_60m": spaces.Box(low=-1, high=np.inf, shape=(self.data_length['bar_60m'], 5), dtype=np.float64),
            # "bar_1d": spaces.Box(low=0, high=np.inf, shape=(self.data_length['bar_1d'], 5), dtype=np.float64),
        })

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

    def _update_subscription(self):
        # update quote subscriptions when underlying_symbol changes
        if self.api.is_changing(self.instrument_quote, "underlying_symbol") or self.target_pos_task is None:
            print("Updating subscription")
            self.underlying_symbol = self.instrument_quote.underlying_symbol
            if self.target_pos_task is not None:
                self.profit += self.target_pos_task.set_target_volume(0, self.instrument_quote.last_price)
            # self.target_pos_task = TargetPosTask(self.api, self.underlying_symbol, offset_priority="昨今开")
            self.target_pos_task = TargetPosTaskOffline()

            self.bar_1s = self.api.get_kline_serial(
                self.underlying_symbol, 1, data_length=self.data_length[Interval.ONE_SEC.value])
            self.bar_1m = self.api.get_kline_serial(
                self.underlying_symbol, 60, data_length=self.data_length[Interval.ONE_MIN.value])
            self.bar_30m = self.api.get_kline_serial(
                self.underlying_symbol, 1800, data_length=self.data_length[Interval.THIRTY_MIN.value])

            # self.bar_1d = self.api.get_kline_serial(
            #     self.underlying_symbol, 86400, data_length=self.data_length['bar_1d'])

    def _reward_function(self):
        # Reward is the profit of the last action
        self.accumulated_reward += self.profit
        return self.profit

    def _get_state(self):
        now = time_to_datetime(self.instrument_quote.datetime)
        while True:
            self.api.wait_update()
            if self.api.is_changing(self.bar_1s.iloc[-1], "datetime"):

                bar_1s = self.bar_1s[self.OHLCV].to_numpy(dtype=np.float64)
                bar_1m = self.bar_1m[self.OHLCV].to_numpy(dtype=np.float64)
                bar_30m = self.bar_30m[self.OHLCV].to_numpy(dtype=np.float64)

                state = dict({
                    "last_volume": np.array([self.last_volume], dtype=np.int64),
                    "hour": np.array([now.hour], dtype=np.int64),
                    "minute": np.array([now.minute], dtype=np.int64),
                    Interval.ONE_SEC.value: bar_1s,
                    Interval.ONE_MIN.value: bar_1m,
                    Interval.THIRTY_MIN.value: bar_30m,
                    # "bar_1d": bar_1d
                })
                if np.isnan(bar_1s).any() or np.isnan(bar_1m).any() or np.isnan(bar_30m).any():
                    self.api.wait_update()
                else:
                    return state

    def step(self, action):
        try:
            assert self.action_space.contains(action)
            action = action[0]
            self.profit = self.target_pos_task.set_target_volume(action, self.instrument_quote.last_price)
            self.api.wait_update()
            self.reward = self._reward_function()
            state = self._get_state()
            self.last_volume = deepcopy(action)
            self.steps += 1

            self._update_subscription()
            self.log_info()
            # wandb.log(self.info)
            if self.steps >= self.max_steps:
                self.done = True
            return state, self.reward, self.done, self.info
        except Exception as e:
            print("Error in step, resetting position to 0")
            self.target_pos_task.set_target_volume(0, self.instrument_quote.last_price)
            self.api.wait_update()
            raise e

    def reset(self):
        """
        Reset the state if a new day is detected.
        """
        print("Resetting")
        self.done = False
        self.steps = 0
        self.last_volume = 0  # last target position volume
        self.last_commision = 0
        self.reward = 0
        self.accumulated_reward = 0
        self.profit = 0
        self.api.wait_update()
        self._update_subscription()
        state = self._get_state()
        self.log_info()
        return state

    def log_info(self,):
        self.info = {
            # "pre_balance": self.account.pre_balance,
            # "static_balance": self.account.static_balance,
            # "balance": self.account.balance,
            # "available": self.account.available,
            # "float_profit": self.account.float_profit,
            # "position_profit": self.account.position_profit,
            # "close_profit": self.account.close_profit,
            # "frozen_margin": self.account.frozen_margin,
            # "margin": self.account.margin,
            # "frozen_commission": self.account.frozen_commission,
            # "commission": self.account.commission,
            # "frozen_premium": self.account.frozen_premium,
            # "premium": self.account.premium,
            # "risk_ratio": self.account.risk_ratio,
            # "market_value": self.account.market_value,
            "training_info/time": time_to_s_timestamp(self.instrument_quote.datetime),
            "training_info/reward": self.reward,
            # "commision_change": self.account.commission - self.last_commision,
            "training_info/last_volume": self.last_volume,
            "training_info/accumulated_reward": self.accumulated_reward,
            "training_info/profit": self.profit,
            "training_info/last_price": self.instrument_quote.last_price,
        }
        # self.last_commision = self.account.commission

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
