import numpy as np
from typing import Dict
import gym
from gym import spaces

state = np.load("./src/state.npy", allow_pickle=True).item()

print(state)


observation_space = observation_space = spaces.Dict({
    "static_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
    "last_volume": spaces.Box(low=-30, high=30, shape=(1,), dtype=np.int64),
    "hour": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int64),
    "minute": spaces.Box(low=0, high=59, shape=(1,), dtype=np.int64),
    "ticks": spaces.Box(low=0, high=np.inf, shape=(30, 8), dtype=np.float64),
    "bar_1m": spaces.Box(low=0, high=np.inf, shape=(30, 5), dtype=np.float64),
    "bar_60m": spaces.Box(low=0, high=np.inf, shape=(30, 5), dtype=np.float64),
    "bar_1d": spaces.Box(low=0, high=np.inf, shape=(30, 5), dtype=np.float64),
})

state = dict({
    "static_balance": state['static_balance'],
    "last_volume": state['last_volume'],
    "hour": state['hour'],
    "minute": state['minute'],
    "ticks": state['ticks'],
    "bar_1m":  state['bar_1m'],
    "bar_60m": state['bar_60m'],
    "bar_1d": state['bar_1d'],
})

if observation_space.contains(state):
    print("Contains")
else:
    print("Not contains")
