from datetime import datetime

import numpy as np

import wandb
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .constant import EnvConfig
from .tools import get_symbols_by_names

from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account
from tqsdk.tafunc import time_to_datetime


class FuturesEnvV1(gym.Env):
    """
    Custom Environment for RL training
    TqApi is required.
    Single symbol and interday only. 
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(gym.Env, self).__init__()
        config: EnvConfig = config['cfg']

        self._skip_env_checking = False
        self._set_config(config)
        self.seed(42)

        self.reset()

    def _update_subscription(self):
        # update quote subscriptions when underlying_symbol changes
        if self.api.is_changing(self.instrument_quote, "underlying_symbol") or self.target_pos_task is None:
            print("Updating subscription")
            self.underlying_symbol = self.instrument_quote.underlying_symbol
            if self.target_pos_task is not None:
                self.target_pos_task.set_target_volume(0)
            self.target_pos_task = TargetPosTask(
                self.api, self.underlying_symbol, offset_priority="昨今开")
            self.ticks = self.api.get_tick_serial(
                self.underlying_symbol, data_length=self.data_length['ticks'])
            self.bar_1m = self.api.get_kline_serial(
                self.underlying_symbol, 60, data_length=self.data_length['bar_1m'])
            self.bar_60m = self.api.get_kline_serial(
                self.underlying_symbol, 3600, data_length=self.data_length['bar_60m'])
            self.bar_1d = self.api.get_kline_serial(
                self.underlying_symbol, 86400, data_length=self.data_length['bar_1d'])

    def _set_config(self, config: EnvConfig):
        # Subscribe instrument quote
        print("Setting config")
        self.api = config.api
        self.account = self.api.get_account()
        # Set target position task
        self.target_pos_task = None
        symbol = get_symbols_by_names(config)[0]
        self.instrument_quote = self.api.get_quote(symbol)

        # Account and API
        self.data_length = config.data_length
        self.underlying_symbol = self.instrument_quote.underlying_symbol
        self.balance = self.account.static_balance

        # RL config
        self.max_steps = config.max_steps
        self.action_space = spaces.Box(
            low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Dict({
            "static_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "last_volume": spaces.Box(low=-config.max_volume, high=-config.max_volume, shape=(1,), dtype=np.int32),
            "hour": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32),
            "minute": spaces.Box(low=0, high=59, shape=(1,), dtype=np.int32),
            "ticks": spaces.Box(low=0, high=np.inf, shape=(self.data_length['ticks'], 8), dtype=np.float32),
            "bar_1m": spaces.Box(low=0, high=np.inf, shape=(self.data_length['bar_1m'], 5), dtype=np.float32),
            "bar_60m": spaces.Box(low=0, high=np.inf, shape=(self.data_length['bar_60m'], 5), dtype=np.float32),
            "bar_1d": spaces.Box(low=0, high=np.inf, shape=(self.data_length['bar_1d'], 5), dtype=np.float32),
        })

    def _reward_function(self):
        # Reward is the change of balance
        pnl = self.account.static_balance - self.balance
        reward = pnl / self.balance
        self.balance = self.account.static_balance
        return reward

    def _get_state(self):
        now = time_to_datetime(self.instrument_quote.datetime)
        static_balance = self.account.static_balance
        ticks = self.ticks[['last_price', 'average', 'volume', 'open_interest', 'ask_price1', 'ask_volume1',
                            'bid_price1', 'bid_volume1', ]].to_numpy()
        bar_1m = self.bar_1m[['open', 'high',
                              'low', 'close', 'volume']].to_numpy()
        bar_60m = self.bar_60m[['open', 'high',
                                'low', 'close', 'volume']].to_numpy()
        bar_1d = self.bar_1d[['open', 'high',
                              'low', 'close', 'volume']].to_numpy()
        return {
            "static_balance": np.array([static_balance]),
            "last_volume": np.array([self.last_volume]),
            "hour": np.array([now.hour]),
            "minute": np.array([now.minute]),
            "ticks": ticks,
            "bar_1m": bar_1m,
            "bar_60m": bar_60m,
            "bar_1d": bar_1d
        }

    def step(self, action: int):
        try:
            assert self.action_space.contains(action)
            self.target_pos_task.set_target_volume(action)
            self.api.wait_update()
            self.reward = self._reward_function()
            state = self._get_state()
            self.last_volume = action
            self.steps += 1

            now = time_to_datetime(self.instrument_quote.datetime)
            self._update_subscription()
            self.log_info(now)
            wandb.log(self.info)
            if self.steps >= self.max_steps:
                self.done = True
            return state, self.reward, self.done, self.info
        except Exception as e:
            self.target_pos_task.set_target_volume(0)
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
        self.reward = 0
        self._update_subscription()
        state = self._get_state()
        now = time_to_datetime(self.instrument_quote.datetime)
        self.log_info(now)
        return state

    def log_info(self, now: datetime):
        self.info = {
            "pre_balance": self.account.pre_balance,
            "static_balance": self.account.static_balance,
            "balance": self.account.balance,
            "available": self.account.available,
            "float_profit": self.account.float_profit,
            "position_profit": self.account.position_profit,
            "close_profit": self.account.close_profit,
            "frozen_margin": self.account.frozen_margin,
            "margin": self.account.margin,
            "frozen_commission": self.account.frozen_commission,
            "commission": self.account.commission,
            "frozen_premium": self.account.frozen_premium,
            "premium": self.account.premium,
            "risk_ratio": self.account.risk_ratio,
            "market_value": self.account.market_value,
            "time": now,
            "reward": self.reward,
        }

    def render(self, mode='human') -> None:
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
