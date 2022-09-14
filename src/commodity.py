import yaml
import os.path

DIR = os.path.dirname(os.path.abspath(__file__))

class Commodity:
    def __init__(self):
        self.filePath = os.path.join(DIR, "commodity.yaml")
        self.commodityConfig = self.loadCommodityConfig(self.filePath)

    def loadCommodityConfig(self, filePath: str):
        with open(filePath, 'r') as stream:
            config = yaml.safe_load(stream)
            # {'auth': {'username': 'xxxx', 'pass': 'xxxxx'}}
            return config['commodity']

    def getCommodityConfig(self, name: str):
        return self.commodityConfig[name]