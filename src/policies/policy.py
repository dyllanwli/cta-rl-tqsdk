from .example.vwap import vwap
from .example.random_forest import random_forest

class Policy:
    def __init__(self, config = None):
        self.config = config

    def load_policy(self, name):
        if name == 'random_forest':
            return random_forest
        elif name == 'vwap':
            return vwap
