from tqsdk import TqApi, TqAuth
from utils import getConfig

class API:

    def __init__(self):
        self.api = self.getAPI()

    def getAPI(self):
        config = getConfig()['auth']
        return TqApi(auth=TqAuth(config['username'], config['pass']))

