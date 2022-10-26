import yaml
import os.path

DIR = os.path.dirname(os.path.abspath(__file__))

class Commodity:
    def __init__(self):
        self.filePath = os.path.join(DIR, "../commodity.yaml")
        self.commodityConfig = self.load_commodity_config(self.filePath)

    def load_commodity_config(self, filePath: str):
        with open(filePath, 'r') as stream:
            config = yaml.safe_load(stream)
            # {'auth': {'username': 'xxxx', 'pass': 'xxxxx'}}
            return config['commodity']

    def get_name(self, name: str):
        return self.commodityConfig[name]

    def get_instrument_name(self, name: str):
        return "KQ.m@" + self.commodityConfig[name]
