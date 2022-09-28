from utils import SETTINGS
from tqsdk import TqApi, TqAuth, TqAccount


class API:
    def __init__(self, account: str = 'a1'):
        self.config = SETTINGS["account"][account]
        self.auth = TqAuth(self.config['username'], self.config['pass'])
        self.live_config = self.config['live_account']
        self.live_account = TqAccount(
            self.live_config['broker_id'],
            self.live_config['account_id'],
            self.live_config['password'])

    def test(self, symbol: str = 'CZCE.CF301'):
        api = TqApi(auth=self.auth)
        quote = api.get_quote(symbol)
        tick = api.get_tick_serial(symbol)
        bar_1m = api.get_kline_serial(symbol, 60)
        bar_5m = api.get_kline_serial(symbol, 300)
        bar_30m = api.get_kline_serial(symbol, 1800)
        bar_1d = api.get_kline_serial(symbol, 86400)
        while True:
            api.wait_update()
            print(quote.datetime, quote.last_price)
