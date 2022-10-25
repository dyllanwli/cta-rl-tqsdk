import numpy as np
from copy import deepcopy
import pytz

import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .constant import EnvConfig
from .tools import DataLoader, get_symbols_by_names, TargetPosTaskOffline
from .factors import Factors

import wandb
from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account, Quote
from tqsdk.tafunc import time_to_datetime, time_to_s_timestamp


class FuturesEnvV3_1(gym.Env):
    """
    Custom Environment for RL training
    TqApi is required.
    Single symbol and interday only. 
    Supported:
        + multiple bar subscription
        + single symbol
        - factors
        + Offline TargetPosTask
        + Offline Data
        + TODO: update dataloader distribution
        + TODO: flatten the observation
    """

    def __init__(self, config):
        super(gym.Env, self).__init__()
        config: EnvConfig = config['cfg']

        self.wandb = config.wandb if config.wandb else False
        if self.wandb:
            wandb.init(project=config.project_name,
                       name=self.wandb, group="train_" + config.interval)

        self._skip_env_checking = True
        self._set_config(config)
        self.seed(42)
        self.reset()

    def _set_config(self, config: EnvConfig):
        """
        Set config for the environment. Only called once in __init__.
        """
        # Subscribe instrument quote
        print("env: Setting config")
        # data config
        self.is_offline = config.is_offline
        self.is_random_sample = config.is_random_sample
        self.dataloader = DataLoader(config)
        self.symbol = get_symbols_by_names(config)[0]
        self.high_freq = config.high_freq
        self.max_hold_steps = config.max_hold_steps
        self.max_steps = config.max_steps
        self.overall_steps = 0

        # Trading config
        self.target_pos_task = TargetPosTaskOffline() if self.is_offline else None
        self.interval: str = config.interval # interval for OHLCV
        self.bar_length: int = 300  # subscribed bar length
        self.ohlcv_length: int = config.data_length[self.interval]   # OHLCV length
        self.factor_length = 20

        # RL config
        self.action_space_type = config.action_space_type
        self.max_action = config.max_volume
        if self.action_space_type == "discrete":
            self.action_space: spaces.Discrete = spaces.Discrete(
                self.max_action*2+1)
        else:
            self.action_space: spaces.Box = spaces.Box(
                low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.int32)

        self.observation_space: spaces.Dict = spaces.Dict({
            "open": spaces.Box(low=0, high=1e10, shape=(self.ohlcv_length,), dtype=np.float32),
            "high": spaces.Box(low=0, high=1e10, shape=(self.ohlcv_length,), dtype=np.float32),
            "low": spaces.Box(low=0, high=1e10, shape=(self.ohlcv_length,), dtype=np.float32),
            "close": spaces.Box(low=0, high=1e10, shape=(self.ohlcv_length,), dtype=np.float32),
            "volume": spaces.Box(low=0, high=1e10, shape=(self.ohlcv_length,), dtype=np.float32),
            "last_action": spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.int32),
            "last_price": spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            "float_profit": spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32),
            # "datetime": spaces.Box(low=0, high=60, shape=(3,), dtype=np.int32),
            ### factors ###
            # "bias": spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32),
            # "macd_bar": spaces.Box(low=-1e10, high=1e10, shape=(self.factor_length, ), dtype=np.float32),
            # "boll": spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "kdj": spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
        })
        self.factors = Factors(self.observation_space, self.factor_length)
        if self.is_offline:
            self._set_offline_data()
            self.balance, self.last_balance = 0, 0
        else:
            self._set_api_data()

    def _set_offline_data(self):
        # get offline data from db
        self.bar_start_step = 0
        offset = self.bar_length + self.bar_start_step + self.max_steps + 10
        self.offline_data: pd.DataFrame = self.dataloader.get_offline_data(
            interval=self.interval, instrument_id=self.symbol, offset=offset)
        # self.offline_data = self.factors.min_max_normalize(self.offline_data)

    def _set_api_data(self):
        self.api: TqApi = self.dataloader.get_api()
        self.account = self.api.get_account()
        self.balance = deepcopy(self.account.balance)
        self.instrument_quote: Quote = self.api.get_quote(self.symbol)
        self.underlying_symbol = self.instrument_quote.underlying_symbol

    def _set_target_volume(self, volume: int):
        if self.is_offline:
            self.profit, self.commission = self.target_pos_task.set_target_volume(
                volume, self.last_price)
        else:
            self.target_pos_task.set_target_volume(volume)
            self.api.wait_update()

    def _update_subscription(self):
        if not self.is_offline:
            self.api.wait_update()
            # update quote subscriptions when underlying_symbol changes
            if self.api.is_changing(self.instrument_quote, "underlying_symbol") or self.target_pos_task is None:
                print("env: Updating subscription")
                self.underlying_symbol = self.instrument_quote.underlying_symbol
                if self.target_pos_task is not None:
                    self._set_target_volume(0)
                self.target_pos_task = TargetPosTask(
                    self.api, self.underlying_symbol, offset_priority="昨今开")
                self.bar_1 = self.api.get_kline_serial(
                    self.underlying_symbol, 1, data_length=self.bar_length)

    def _reward_function(self):
        # Reward is the profit of the last action
        # set reward bound to [-1, 1] using tanh
        # hold_penalty = 0.0001
        # reward = np.tanh((self.profit - hold_penalty)/100)
        reward = self.profit
        self.accumulated_profit += self.profit
        self.balance += self.profit
        # self.accumulated_reward += reward
        if self.profit == 0:
            self.holded_steps += 1
        return reward

    def _get_state(self):
        """
        Get state from data
        """
        if self.is_offline:
            # offline state0
            self.bar_1 = self.offline_data.iloc[self.bar_start_step:
                                                self.bar_start_step+self.bar_length]
            self.last_price = self.bar_1.iloc[-1]['close']
            self.volume = self.bar_1.iloc[-1]['volume']
            self.last_datatime = self.bar_1.iloc[-1]['datetime']

            ohlcv = self.bar_1[self.factors.OHLCV].iloc[-self.ohlcv_length:].to_numpy(
                dtype=np.float32)
        else:
            # online state
            while True:
                self.api.wait_update()
                if self.api.is_changing(self.bar_1.iloc[-1], "datetime"):
                    ohlcv = self.bar_1[self.factors.OHLCV].iloc[-self.ohlcv_length:].to_numpy(
                        dtype=np.float32)
                    self.last_price = self.instrument_quote.last_price
                    self.volume = self.instrument_quote.volume
                    self.last_datatime = self.instrument_quote.datetime
                    if np.isnan(ohlcv).any():
                        self.api.wait_update()
                    else:
                        break

        state = dict()
        # datetime_state = time_to_datetime(self.last_datatime).astimezone(pytz.timezone('Asia/Shanghai'))
        state['last_action'] = np.array([self.last_action], dtype=np.int32)
        state["last_price"] = np.array([self.last_price], dtype=np.float32)
        state["float_profit"] = np.array([self.accumulated_profit], dtype=np.float32)
        # state["datetime"] = np.array([datetime_state.month, datetime_state.hour, datetime_state.minute], dtype=np.int32)

        ohlcv_state = self.factors.set_ohlcv_state(ohlcv) 
        state.update(ohlcv_state)
        factors_state, self.factors_info = self.factors.set_state_factors(bar_data=self.bar_1, last_price=self.last_price)
        state.update(factors_state)
        return state
    
    def _preprocess_action(self, action):
        assert self.action_space.contains(action)
        if self.action_space_type == "discrete":
            action = int(action) - 1
        else:
            action = int(action[0])
        if self.high_freq and action == 0 and self.profit != 0:
            # end episode if all postiions are closed
            self.done = True
        return action


    def reset(self):
        """
        Reset the state if a new day is detected.
        """
        self.done = False
        self.steps = 0
        self.last_action = 0
        self.holded_steps = 0
        self.last_commision, self.commission = 0, 0

        self.reward, self.profit = 0, 0
        # self.accumulated_reward = 0
        self.accumulated_profit = 0

        self._update_subscription()
        if self.is_random_sample:
            self._set_offline_data()
        state = self._get_state()
        return state

    def step(self, action):
        try:
            if self.steps >= self.max_steps or (self.high_freq and self.holded_steps >= self.max_hold_steps):
                self.done = True
                self._set_target_volume(0)
                self.last_action = 0
            else:
                action = self._preprocess_action(action)
                self._set_target_volume(action)
                self.last_action = action
            self.reward = self._reward_function()
            state = self._get_state()
            self.steps += 1
            self.overall_steps += 1
            self.bar_start_step += 1
            self._update_subscription()
            self.log_info()
            return state, self.reward, self.done, self.info
        except Exception as e:
            print("env: Error in step, resetting position to 0. Action: ", action)
            self._set_target_volume(0)
            raise e

    ##### Logging functions #####
    def log_epsoide_info(self):
        if self.wandb:
            wandb.log(
                {"training_info/accumulated_profit": self.accumulated_profit,
                #  "training_info/accumulated_reward": self.accumulated_reward,
                 "training_info/holded_steps": self.holded_steps, })

    def log_info(self,):
        if self.wandb:
            if self.is_offline:
                self.info = {
                    # "training_info/time": time_to_s_timestamp(self.last_datatime),
                    "training_info/last_action": self.last_action,
                }
            else:
                self.info = {
                    "backtest_info/pre_balance": self.account.pre_balance,
                    "backtest_info/static_balance": self.account.static_balance,
                    "backtest_info/balance": self.account.balance,
                    "backtest_info/available": self.account.available,
                    "backtest_info/float_profit": self.account.float_profit,
                    "backtest_info/position_profit": self.account.position_profit,
                    "backtest_info/close_profit": self.account.close_profit,
                    # "backtest_info/frozen_margin": self.account.frozen_margin,
                    # "backtest_info/margin": self.account.margin,
                    # "backtest_info/frozen_commission": self.account.frozen_commission,
                    "backtest_info/commission": self.account.commission,
                    # "backtest_info/frozen_premium": self.account.frozen_premium,
                    # "backtest_info/premium": self.account.premium,
                    "backtest_info/risk_ratio": self.account.risk_ratio,
                    # "backtest_info/market_value": self.account.market_value,
                    "backtest_info/time": time_to_s_timestamp(self.last_datatime),
                    "backtest_info/commision_change": self.account.commission - self.last_commision,
                    "backtest_info/last_action": self.last_action,
                    "backtest_info/volume": self.instrument_quote.volume,
                }
            self.info["factors/last_price"] = self.last_price
            self.info.update(self.factors_info)
            if self.done:
                self.log_epsoide_info()
            if self.overall_steps % self.max_steps == 0:
                self.info["training_info/daily_profit"] = self.balance - self.last_balance
                self.last_balance = self.balance
            wandb.log(self.info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
