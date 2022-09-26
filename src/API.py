from utils import SETTINGS
from tqsdk import TqApi, TqAuth


class API:
    def __init__(self, account: str = 'a1'):
        self.config = SETTINGS["account"][account]
        self.auth = TqAuth(self.config['username'], self.config['pass'])
