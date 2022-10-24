from collections import deque
from copy import deepcopy
import pandas as pd
from datetime import date, datetime

from tqsdk.tafunc import time_to_s_timestamp, time_to_datetime

import numpy as np
import gym
from gym import spaces

# import logging 
# class TargetPosTaskOffline:
#     def __init__(self, commission: float = 5.0, verbose: int = 0):
#         self.last_volume = 0
#         self.positions = deque([])
#         self.commission = commission
#         self.margin_rate = 4.0
#         if verbose == 0:
#             logging.basicConfig(level=logging.DEBUG)
#         else:
#             logging.basicConfig(level=logging.INFO)
    
#     def insert_order(self,):
#         # TODO insert order
#         pass

#     def set_target_volume(self, volume, price: float):
#         profit = 0
#         if self.last_volume == volume:
#             logging.debug("hold position")
#             return 0
#         if volume > 0 and self.last_volume >= 0:
#             position_change = volume - self.last_volume
#             if position_change > 0:
#                 logging.debug("buy long %d", position_change)
#                 for _ in range(position_change):
#                     self.positions.append(price)
#             else:
#                 logging.debug("sell long %d", position_change)
#                 for _ in range(-position_change):
#                     profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
#         elif volume < 0 and self.last_volume <= 0:
#             position_change = self.last_volume - volume
#             if position_change > 0:
#                 logging.debug("buy short %d", position_change)
#                 for _ in range(position_change):
#                     self.positions.append(price)
#             else:
#                 logging.debug("sell short %d", position_change)
#                 for _ in range(-position_change):
#                     profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
#         elif volume >= 0 and self.last_volume < 0:
#             logging.debug("sell all short %d", self.last_volume)
#             while len(self.positions) > 0:
#                 profit += (self.positions.popleft() - price) * self.margin_rate - self.commission
#             logging.debug("buy long %d", volume)
#             for _ in range(volume):
#                 self.positions.append(price)
#         elif volume <= 0 and self.last_volume > 0:
#             logging.debug("sell all long %d", self.last_volume)
#             while len(self.positions) > 0:
#                 profit += (price - self.positions.popleft()) * self.margin_rate - self.commission
#             logging.debug("buy short %d", volume)
#             for _ in range(-volume):
#                 self.positions.append(price)
#         self.last_volume = volume
#         return profit

# c = [12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565,
#      12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565,
#      12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565,
#      12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565, 12565,
#      12565, 12565, 12565, 12565, 12565, 12470, 12470, 12410, 12420, 12400,
#      12400, 12435, 12455, 12440, 12475, 12495, 12525, 12520, 12525, 12525,
#      12500, 12505, 12520, 12500, 12505, 12555, 12615, 12610, 12605, 12590,
#      12590, 12600, 12600, 12595, 12605, 12625, 12615, 12605, 12615, 12615,
#      12660, 12655, 12680, 12665, 12655, 12660, 12685, 12670, 12635, 12635,
#      12640, 12635, 12615, 12615, 12635, 12665, 12650, 12645, 12660, 12655,
#      12640, 12660, 12660, 12650, 12665, 12655, 12660, 12655, 12685, 12715,
#      12715, 12755, 12800, 12765, 12780, 12745, 12735, 12735, 12740, 12700,]

# c = [11000, 11010, 11000, 11020, 11030, 11040, 11020, 11000, 11000, 11000,]
# target_pos_task = TargetPosTaskOffline()
# actions = [1, 0, 0, 1, 0, 0, -1, 1, 0, 0]
# p = 0
# for a, i in zip(actions, c):
#      # action: 1,0,-1
#      profit = target_pos_task.set_target_volume(a, i)
#      print(profit)
#      p += profit
# print(p)

# offset = 300

# start_dt = datetime(2022, 1, 1, 0, 0, 0)
# end_dt = datetime(2022, 2, 1, 0, 0, 0)

# low_dt = time_to_s_timestamp(start_dt)
# high_dt = time_to_s_timestamp(end_dt) - offset - 30

# sample_start = np.random.randint(low = low_dt, high = high_dt, size=1)[0]

# print(sample_start)

# dt = time_to_datetime(sample_start)
# print(dt)

normalized_cols = ["high", "open", "low", "close", "volume", "open_oi", "close_oi"]
# raw_cols = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_volume", "raw_open_oi", "raw_close_oi"]
df = pd.read_csv('test.csv')
ndarray = df[normalized_cols].iloc[-5:].to_numpy(dtype=np.float32)
high = ndarray[:, 0]
open = ndarray[:, 1]
low = ndarray[:, 2]
close = ndarray[:, 3]
volume = ndarray[:, 4]
open_oi = ndarray[:, 5]
close_oi = ndarray[:, 6]
print(high)
print(open)
print(low)
