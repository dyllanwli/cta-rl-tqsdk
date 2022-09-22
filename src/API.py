from utils import get_config
from tqsdk import TqApi, TqAuth


class API:
    def __init__(self, account: str = 'a1'):
        self.config = get_config()[account]
        self.auth = TqAuth(self.config['username'], self.config['pass'])