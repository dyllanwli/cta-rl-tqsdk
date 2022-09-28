import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from constant import EnvConfig
from commodity import Commodity

from tqsdk import InsertOrderUntilAllTradedTask, TqBacktest, TqSim, TqApi, TqAccount


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
        self.bar_1m = self.api.get_kline_serial(
            self.underlying_symbol, 60, data_length=self.data_length['bar_1m'])
        self.bar_5m = self.api.get_kline_serial(
            self.underlying_symbol, 300, data_length=self.data_length['bar_5m'])
        
        self.ticks = self.api.get_tick_serial(self.underlying_symbol, data_length=self.data_length['ticks'])


    def _set_config(self, config: EnvConfig):
        cmod = Commodity()
        self.symbol = cmod.get_instrument_name(config.symbols[0])
        self.instrument_quote = self.api.get_quote(self.symbol)

        self.data_length = config.data_length

        self.underlying_symbol = self.instrument_quote.underlying_symbol
        self.action_space = spaces.Box(
            low=-config.max_volume, high=config.max_volume, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Dict({
            "curremt_volume": spaces.Box(low=-config.max_volume, high=-config.max_volume, shape=(1,), dtype=np.int32),
            "OHLCV": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
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

    def insert_order(self, action):
        pass

    def step(self, action):
        insert_order = InsertOrderUntilAllTradedTask(
            self.api, self.symbol, action)
        self.api.wait_update()

    def reset(self):
        """
        Reset the state if a new day is detected.
        """

    def render(self, mode='human') -> None:
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()
