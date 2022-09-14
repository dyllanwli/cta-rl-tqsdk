from utils import getConfig
from tqsdk import TqApi, TqAuth


class API:
    def __init__(self, account: str = 'a1'):
        self.config = getConfig()[account]
        self.auth = TqAuth(self.config['username'], self.config['pass'])
        self.api = TqApi(auth=self.auth)