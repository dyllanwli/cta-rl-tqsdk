import gym
from gym import error, spaces, utils
from gym.utils import seeding

from constant import EnvConfig

class FuturesEnvV1(gym.Env):
    """
    Custom Environment with Tqsdk2 for RL training
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, config: EnvConfig):
        super(gym.Env, self).__init__()

        self._set_config(config)
        self.seed(42)

        self.reset()
    
    def _set_config(self, config: EnvConfig):
        

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human') -> None:
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        return super().close()