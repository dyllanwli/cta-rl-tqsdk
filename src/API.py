from utils import SETTINGS
from tqsdk import TqApi, TqAuth


class API:
    def __init__(self, account: str = 'a1'):
        self.config = SETTINGS["account"][account]
        self.auth = TqAuth(self.config['username'], self.config['pass'])

    def test(self, symbol: str = 'SHFE.ag2212'):
        api = TqApi(auth=self.auth)
        quote = api.get_quote(symbol)
        while True:
            api.wait_update()
            print(quote.datetime, quote.last_price)
