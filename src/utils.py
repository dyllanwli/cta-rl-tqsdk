import yaml
import os.path

DIR = os.path.dirname(os.path.abspath(__file__))


def get_config():
    # Read YAML file
    configFilePath = os.path.join(DIR, "config.yaml")
    with open(configFilePath, 'r') as stream:
        config = yaml.safe_load(stream)
        # {'auth': {'username': 'xxxx', 'pass': 'xxxxx'}}
        return config['account']



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