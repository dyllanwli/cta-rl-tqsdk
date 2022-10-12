from collections import deque
import pandas as pd
from datetime import date, datetime

from tqsdk.tafunc import time_to_s_timestamp, time_to_datetime

import numpy as np
import gym
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

# if action_space.contains(np.array([[-20]])):
#     print("yes")
class TargetPosTaskOffline:
    def __init__(self, commission: float = 1.0):
        self.last_volume = 0
        self.positions = deque([])
        self.commission = commission
        self.margin_rate = 1.0

    def set_target_volume(self, volume, price: float):
        profit = 0
        if self.last_volume == volume:
            return
        if volume * self.last_volume > 0:
            position_change = volume - self.last_volume
            sign = 1 if volume > 0 else -1
            if position_change > 0:
                print("buy", position_change)
                for _ in range(position_change):
                    self.positions.append(price)
            else:
                print("sell", position_change)
                for _ in range(-position_change):
                    profit += sign * (price - self.positions.popleft()) * self.margin_rate - self.commission
        else:
            if volume >= 0:
                print("sell short", self.last_volume)
                for _ in range(-self.last_volume):
                    profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
                print("buy long", volume)
                for _ in range(volume):
                    self.positions.append(price)
            else:
                print("sell long", self.last_volume)
                for _ in range(self.last_volume):
                    profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
                print("buy short", volume)
                for _ in range(-volume):
                    self.positions.append(price)
        self.last_volume = volume
        return profit

# target_pos_task = TargetPosTaskOffline()


max_steps = 40000 # seconds
start_dt = date(2016, 1, 1)
end_dt = date(2022, 9, 1)

start_dt = datetime.combine(start_dt, datetime.min.time())
end_dt = datetime.combine(end_dt, datetime.max.time())

print("start_dt", start_dt)

low_dt = time_to_s_timestamp(start_dt)
high_dt = time_to_s_timestamp(end_dt) - max_steps

sample_start = np.random.randint(low = low_dt, high = high_dt, size=1)[0]
sample_end = sample_start + max_steps

start_dt = time_to_datetime(sample_start)
end_dt = time_to_datetime(sample_end)


print("start_dt", start_dt.date())
print("end_dt", end_dt)