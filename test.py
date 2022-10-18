from collections import deque
import pandas as pd
from datetime import date, datetime

# from tqsdk.tafunc import time_to_s_timestamp, time_to_datetime

import numpy as np
import gym
from gym import spaces
# from gym import spaces
# action_space : spaces.Box = spaces.Box(
#             low=-30, high=30, shape=(1,), dtype=np.int64)
# observation = spaces.Dict({
#             "last_volume": spaces.Box(low=-30, high=30, shape=(1,), dtype=np.int64),
#             "hour": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int64),
#             "minute": spaces.Box(low=0, high=59, shape=(1,), dtype=np.int64),
#             "bar_1s": spaces.Box(low=-1, high=np.inf, shape=(2, 5), dtype=np.float64),
#             "bar_1m": spaces.Box(low=-1, high=np.inf, shape=(2, 5), dtype=np.float64),
#             "bar_30m": spaces.Box(low=-1, high=np.inf, shape=(2, 5), dtype=np.float64),
#             # "bar_60m": spaces.Box(low=-1, high=np.inf, shape=(self.data_length['bar_60m'], 5), dtype=np.float64),
#             # "bar_1d": spaces.Box(low=0, high=np.inf, shape=(self.data_length['bar_1d'], 5), dtype=np.float64),
#         })

# state = dict({'last_volume': np.array([-19]),
#               'hour': np.array([7]),
#               'minute': np.array([2]),
#               'bar_1s': np.array([[11260., 11260., 11260., 11260.,     0.], [11260., 11260., 11260., 11260.,     0.]]),
#               'bar_1m': np.array([[11265., 11270., 11260., 11270.,   534.], [11270., 11270., 11270., 11270.,     0.]]),
#               'bar_30m': np.array([[11300., 11315., 11270., 11280.,  7016.], [11275., 11275., 11275., 11275.,     0.]])})


# if observation.contains(state):
#     print("yes")
from typing import NamedTuple

class InitOverallStep(NamedTuple):
    ONE_SEC: int = 2*60*60
    FIVE_SEC: int = 2*60*12
    ONE_MIN: int = 2*60

init_over_step = InitOverallStep()

print(InitOverallStep.ONE_SEC)