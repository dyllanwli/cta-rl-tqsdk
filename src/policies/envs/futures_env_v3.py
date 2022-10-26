from datetime import datetime
import numpy as np
from copy import deepcopy

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .constant import EnvConfig
from .tools import get_symbols_by_names, TargetPosTaskOffline
from .factors import Factors

import wandb
from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account, Quote
from tqsdk.tafunc import time_to_datetime, time_to_s_timestamp

from src.utils.utils import Interval


class FuturesEnvV3(gym.Env):
    """
    Custom Environment for RL training
    TqApi is required.
    Single symbol and interday only. 
    Supported:
        + multiple bar subscription
        + single symbol
        - factors
        + Offline TargetPosTask
        - Offline Data
    """

    def __init__(self, config):
        super(gym.Env, self).__init__()
        config: EnvConfig = config['cfg']

        self.wandb = config.wandb if config.wandb else False
        if self.wandb:
            wandb.init(project="futures-trading", name = self.wandb)

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
        self.factors = Factors()

        # Account and API
        self.data_length = config.data_length
        self.underlying_symbol = self.instrument_quote.underlying_symbol
        self.balance = deepcopy(self.account.balance)

        # RL config
        self.max_steps = config.max_steps
        self.action_space: spaces.Box = spaces.Box(
            low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int32)

        self.factor_length = 50
        self.observation_space: spaces.Dict = spaces.Dict({
            # "last_volume": spaces.Box(low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int32),
            "last_price": spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            # "hour": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32),
            # "minute": spaces.Box(low=0, high=59, shape=(1,), dtype=np.int32),
            Interval.ONE_SEC.value: spaces.Box(low=0, high=1e10, shape=(self.data_length[Interval.ONE_SEC.value], 5), dtype=np.float32),
            # Interval.ONE_MIN.value: spaces.Box(low=0, high=1e10, shape=(self.data_length[Interval.ONE_MIN.value], 5), dtype=np.float32),
            # Interval.THIRTY_MIN.value: spaces.Box(low=0, high=1e10, shape=(self.data_length[Interval.THIRTY_MIN.value], 5), dtype=np.float32),
            "macd_bar": spaces.Box(low=-np.inf, high=np.inf, shape=(self.factor_length, ), dtype=np.float32),
            "rsi": spaces.Box(low=-np.inf, high=np.inf, shape=(self.factor_length,), dtype=np.float32),
        })

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

    def _update_subscription(self):
        # update quote subscriptions when underlying_symbol changes
        if self.api.is_changing(self.instrument_quote, "underlying_symbol") or self.target_pos_task is None:
            print("Updating subscription")
            self.underlying_symbol = self.instrument_quote.underlying_symbol
            if self.target_pos_task is not None:
                self.profit += self.target_pos_task.set_target_volume(
                    0, self.instrument_quote.last_price)
            self.target_pos_task = TargetPosTaskOffline()

            self.bar_1s = self.api.get_kline_serial(
                self.underlying_symbol, 1, data_length=1000)


    def _reward_function(self):
        # Reward is the profit of the last action
        self.accumulated_reward += self.profit
        return self.profit

    def _get_state(self):
        # now = time_to_datetime(self.instrument_quote.datetime)
        while True:
            self.api.wait_update()
            if self.api.is_changing(self.bar_1s.iloc[-1], "datetime"):

                bar_1s = self.bar_1s[self.OHLCV].iloc[-self.data_length[Interval.ONE_SEC.value]:].to_numpy(dtype=np.float32)

                state = dict({
                    # "last_volume": np.array([self.last_volume], dtype=np.int32),
                    "last_price": np.array([self.instrument_quote.last_price], dtype=np.float32),
                    Interval.ONE_SEC.value: bar_1s,
                    "rsi": np.array(self.factors.rsi(self.bar_1s[-self.factor_length:], n=30)[-self.factor_length:], dtype=np.float32),
                    "macd_bar": np.array(self.factors.macd_bar(self.bar_1s.iloc[-self.factor_length:], short=60, long=120, m=30), dtype=np.float32),
                })
                if np.isnan(bar_1s).any():
                    self.api.wait_update()
                else:
                    return state

    def step(self, action):
        try:
            assert self.action_space.contains(action)
            action = action[0]
            self.profit = self.target_pos_task.set_target_volume(
                action, self.instrument_quote.last_price)
            self.api.wait_update()
            self.reward = self._reward_function()
            state = self._get_state()
            self.last_volume = action
            self.steps += 1

            self._update_subscription()
            self.log_info()
            if self.steps >= self.max_steps:
                self.done = True
                if self.wandb:
                    wandb.log({"training_info/accumulated_reward": self.accumulated_reward})
            return state, self.reward, self.done, self.info
        except Exception as e:
            print("Error in step, resetting position to 0")
            self.target_pos_task.set_target_volume(
                0, self.instrument_quote.last_price)
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
            # "commision_change": self.account.commission - self.last_commision,
            "training_info/last_volume": self.last_volume,
            "training_info/profit": self.profit,
            "training_info/last_price": self.instrument_quote.last_price,
        }
        # self.last_commision = self.account.commission
        if self.wandb:
            wandb.log(self.info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
