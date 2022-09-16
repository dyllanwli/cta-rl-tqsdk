class Policy:
    def __init__(self, config = None):
        self.config = config

    def load_policy(self, name):
        if name == 'random_forest':
            from .example.random_forest import random_forest
            return random_forest
        elif name == 'vwap':
            from .example.vwap import VWAP
            return VWAP()
        elif name == 'r_breaker':
            from .example.r_breaker import RBreaker
            return RBreaker()
