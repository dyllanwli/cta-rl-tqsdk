from datetime import datetime

import numpy as np
from copy import deepcopy
import wandb
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .constant import EnvConfig
from .tools import get_symbols_by_names

from tqsdk import TargetPosTask, TqSim, TqApi, TqAccount
from tqsdk.objs import Account
from tqsdk.tafunc import time_to_datetime


class FuturesEnvV3(gym.Env):
    """
    Custom Environment for RL training
    TqApi is required.
    Single symbol and interday only. 
    """
    def __init__(self, config):
        super(gym.Env, self).__init__()
        config: EnvConfig = config['cfg']
        wandb.init(project="futures-trading", name="_" +
                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

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
        self.target_pos_task = None
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
            "bar_1s": spaces.Box(low=-1, high=np.inf, shape=(self.data_length['bar_1s'], 5), dtype=np.float64),
            "bar_1m": spaces.Box(low=-1, high=np.inf, shape=(self.data_length['bar_1m'], 5), dtype=np.float64),
            "bar_30m": spaces.Box(low=-1, high=np.inf, shape=(self.data_length['bar_30m'], 5), dtype=np.float64),
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
                self.target_pos_task.set_target_volume(0)
            self.target_pos_task = TargetPosTask(
                self.api, self.underlying_symbol, offset_priority="昨今开")

            self.bar_1s = self.api.get_kline_serial(
                self.underlying_symbol, 1, data_length=self.data_length['bar_1s'])
            self.bar_1m = self.api.get_kline_serial(
                self.underlying_symbol, 60, data_length=self.data_length['bar_1m'])
            self.bar_30m = self.api.get_kline_serial(
                self.underlying_symbol, 1800, data_length=self.data_length['bar_30m'])

            # self.bar_1d = self.api.get_kline_serial(
            #     self.underlying_symbol, 86400, data_length=self.data_length['bar_1d'])

    def _reward_function(self):
        # Reward is the change of balance
        pnl = self.account.balance - self.balance
        reward = pnl / self.balance
        self.balance = deepcopy(self.account.balance)

        self.accumulated_reward += reward
        return reward

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
                    "bar_1s": bar_1s,
                    "bar_1m": bar_1m,
                    "bar_30m": bar_30m,
                    # "bar_1d": bar_1d
                })
                if np.isnan(bar_1s).any() or np.isnan(bar_1m).any() or np.isnan(bar_30m).any():
                    print("Nan in state, waiting for new data")
                    self.api.wait_update()
                else:
                    return state

    def step(self, action):
        try:
            assert self.action_space.contains(action)
            action = action[0]
            self.target_pos_task.set_target_volume(action)
            self.api.wait_update()
            self.reward = self._reward_function()
            state = self._get_state()
            self.last_volume = deepcopy(action)
            self.steps += 1

            self._update_subscription()
            self.log_info()
            wandb.log(self.info)
            if self.steps >= self.max_steps:
                self.done = True
            return state, self.reward, self.done, self.info
        except Exception as e:
            print("Error in step, resetting position to 0")
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
        self.last_commision = 0
        self.reward = 0
        self.accumulated_reward = 0
        self.api.wait_update()
        self._update_subscription()
        state = self._get_state()
        self.log_info()
        # np.save("state.npy", state) # debug
        return state

    def log_info(self,):
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
            # "frozen_premium": self.account.frozen_premium,
            # "premium": self.account.premium,
            "risk_ratio": self.account.risk_ratio,
            # "market_value": self.account.market_value,
            "time": self.instrument_quote.datetime,
            "reward": self.reward,
            "commision_change": self.account.commission - self.last_commision,
            "last_volume": self.last_volume,
        }
        self.last_commision = self.account.commission

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
