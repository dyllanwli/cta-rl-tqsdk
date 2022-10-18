
from enum import Enum
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



class Interval(Enum):
    ONE_SEC = "1s"
    FIVE_SEC = "5s"
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    TICK = "tick"

class MaxStepByDay(Enum):
    ONE_SEC = 13500
    FIVE_SEC = 4320
    ONE_MIN = 360
    FIVE_MIN = 72
    FIFTEEN_MIN = 24



SETTINGS = get_config()