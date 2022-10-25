
from typing import NamedTuple
import yaml
import os.path


DIR = os.path.dirname(os.path.abspath(__file__))

def get_config() -> dict:
    # Read YAML file
    configFilePath = os.path.join(DIR, "config.yaml")
    with open(configFilePath, 'r') as stream:
        config = yaml.safe_load(stream)
        return config


def get_trading_time(quote):
    """
    quote: quote object
    """
    # Get trading time
    tradingTime = quote['trading_time']
    day = tradingTime['day']
    night = tradingTime['night']
    # if it's overnight, it will be over 24:00
    # e.g. 21:00:00 - 26:30:00 means 21:00:00 - 2:30:00 next day
    return day, night



class Interval(NamedTuple):
    ONE_SEC: str = "1s"
    FIVE_SEC: str = "5s"
    ONE_MIN: str = "1m"
    FIVE_MIN: str = "5m"
    FIFTEEN_MIN: str = "15m"
    THIRTY_MIN: str = "30m"
    ONE_HOUR: str = "1h"
    FOUR_HOUR: str = "4h"
    ONE_DAY: str = "1d"
    TICK: str = "tick"

class InitOverallStep(NamedTuple):
    ONE_SEC: int = 2*60*60
    FIVE_SEC: int = 2*60*12
    ONE_MIN: int = 2*60

max_step_by_day = {
    "1s": 20700,
    "5s": 4140,
    "1m": 345,
    "5m": 72,
    "15m": 24,
}


SETTINGS = get_config()