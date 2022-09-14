import yaml
import os.path

DIR = os.path.dirname(os.path.abspath(__file__))


def getConfig():
    # Read YAML file
    configFilePath = os.path.join(DIR, "config.yaml")
    with open(configFilePath, 'r') as stream:
        config = yaml.safe_load(stream)
        # {'auth': {'username': 'xxxx', 'pass': 'xxxxx'}}
        return config['account']



def getTradingTime(quote):
    """
    quote: quote object
    """
    # Get trading time
    tradingTime = quote['trading_time']
    day = tradingTime['day']
    night = tradingTime['night']
    return day, night