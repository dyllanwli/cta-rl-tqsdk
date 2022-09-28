from datetime import datetime

import numpy as np

import wandb
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .constant import EnvConfig
from .tools import get_symbol_by_name

from tqsdk import TargetPosTask, TqBacktest, TqSim, TqApi, TqAccount
from tqsdk.objs import Account
from tqsdk.tafunc import time_to_datetime



class FuturesEnvV1(gym.Env):
    """
    Custom Environment for RL training
    TqApi is required.
    Single symbol and interday only. 
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: EnvConfig):
        super(gym.Env, self).__init__()

        self._set_config(config)
        self.seed(42)

        self.reset()

    def _update_subscription(self):
        self.underlying_symbol = self.instrument_quote.underlying_symbol
        self.ticks = self.api.get_tick_serial(
            self.underlying_symbol, data_length=self.data_length['ticks'])
        self.bar_1m = self.api.get_kline_serial(
            self.underlying_symbol, 60, data_length=self.data_length['bar_1m'])
        self.bar_60m = self.api.get_kline_serial(
            self.underlying_symbol, 3600, data_length=self.data_length['bar_60m'])
        self.bar_1d = self.api.get_kline_serial(
            self.underlying_symbol, 86400, data_length=self.data_length['bar_1d'])

    def _set_config(self, config: EnvConfig):
        self.symbol = get_symbol_by_name(config.symbols[0])
        self.instrument_quote = self.api.get_quote(self.symbol)

        self.data_length = config.data_length

        self.underlying_symbol = self.instrument_quote.underlying_symbol
        self.action_space = spaces.Box(
            low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Dict({
            "curremt_volume": spaces.Box(low=-config.max_volume, high=-config.max_volume, shape=(1,), dtype=np.int32),
            "hour": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32),
            "minute": spaces.Box(low=0, high=59, shape=(1,), dtype=np.int32),
            "ticks": spaces.Box(low=0, high=np.inf, shape=(9, self.data_length['ticks']), dtype=np.float32),
            "bar_1m": spaces.Box(low=0, high=np.inf, shape=(5, self.data_length['bar_1m']), dtype=np.float32),
            "bar_60m": spaces.Box(low=0, high=np.inf, shape=(5, self.data_length['bar_60m']), dtype=np.float32),
            "bar_1d": spaces.Box(low=0, high=np.inf, shape=(5, self.data_length['bar_1d']), dtype=np.float32),
        })

    def _set_account(self, config: EnvConfig):
        """
        Set account and API for TqApi
        """
        if config.backtest:
            # backtest
            print("backtest mode")
            self.account = TqSim(init_balance=config.init_balance)
            self.api = TqApi(auth=config.auth, backtest=config.backtest,
                             account=self.account)
        else:
            # live or sim
            if config.live_market:
                print("live market mode")
                self.api = TqApi(config.live_account, auth=config.auth)
                self.account = self.api.get_account()
            else:
                print("sim mode")
                self.account = TqSim(init_balance=config.init_balance)
                self.api = TqApi(auth=config.auth)

    def get_account_info(self):
        pass
    
    def _reward_function(self):
        pass

    def step(self, action):
        self.api.wait_update()

    def reset(self):
        """
        Reset the state if a new day is detected.
        """
        self._update_subscription()
        now = time_to_datetime(self.instrument_quote.datetime)
        self.wandb_log(self.account, now)
    

    def wandb_log(self, account: Account, now: datetime):
        wandb.log({
            # "currency": account.currency,
            "pre_balance": account.pre_balance,
            "static_balance": account.static_balance,
            "balance": account.balance,
            "available": account.available,
            "float_profit": account.float_profit,
            "position_profit": account.position_profit,
            "close_profit": account.close_profit,
            "frozen_margin": account.frozen_margin,
            "margin": account.margin,
            "frozen_commission": account.frozen_commission,
            "commission": account.commission,
            "frozen_premium": account.frozen_premium,
            "premium": account.premium,
            "risk_ratio": account.risk_ratio,
            "market_value": account.market_value,
            "time": now
        })

    def render(self, mode='human') -> None:
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
