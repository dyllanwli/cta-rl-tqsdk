import copy
import weakref
from collections.abc import MutableMapping
from typing import List, Dict
from datetime import date

from tqsdk import TqAuth, TqSim, TqBacktest, BacktestFinished, TqAccount


class Entity(MutableMapping):
    def _instance_entity(self, path):
        self._path = path
        self._listener = weakref.WeakSet()

    def __setitem__(self, key, value):
        return self.__dict__.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__dict__.__delitem__(key)

    def __getitem__(self, key):
        return self.__dict__.__getitem__(key)

    def __iter__(self):
        return iter({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def __len__(self):
        return len({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def __str__(self):
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def __repr__(self):
        return '{}, D({})'.format(super(Entity, self).__repr__(),
                                  {k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def copy(self):
        return copy.copy(self)

class EnvConfig(Entity):
    # env config is a entity
    def __init__(self, 
        auth: TqAuth,
        symbols: List[str],
        backtest: TqBacktest = None,
        live_market: bool = False,
        live_account: TqAccount = None,
    ):
        self.auth: TqAuth = auth
        self.symbols: List[str] = symbols
        self.backtest: TqBacktest = backtest
        self.live_market: bool = live_market
        self.live_account: TqAccount = live_account

        # other config params
        self.init_balance: float = 100000
        self.max_volume: int = 30
        self.trade_position_ratio_limit: float = 0.9

        self.data_length: Dict[str, int] = {
            "ticks": 200,
            "bar_1m": 200,
            "bar_5m": 200,
            "bar_30m": 200,
            "bar_60m": 200,
            "bar_1d": 200,
        }






