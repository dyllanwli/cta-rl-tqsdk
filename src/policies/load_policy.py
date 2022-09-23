class LoadPolicy:
    def __init__(self, config=None):
        self.config = config

    def load_policy(self, name):
        if name == 'random_forest':
            from .example.random_forest import RandomForest
            return RandomForest()
        elif name == 'vwap':
            from .example.vwap import VWAP
            return VWAP()
        elif name == 'r_breaker':
            from .example.r_breaker import RBreaker
            return RBreaker()
        elif name == 'r_breaker_overnight':
            from .example.r_breaker_overnight import RBreakerOvernight
            return RBreakerOvernight()
        elif name == 'grid':
            from .example.grid import Grid
            return Grid()
