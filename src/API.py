from utils import SETTINGS
from tqsdk import TqApi, TqAuth, TqAccount


class API:
    def __init__(self, account: str = 'a1', live_account: str = None):
        account_settings = SETTINGS['account'][account]
        self.auth = TqAuth(
            account_settings['username'], account_settings['pass'])

        if live_account is not None:
            live_account_settings = SETTINGS['live_account'][live_account]
            self.live_account = TqAccount(
                live_account_settings['broker_id'],
                live_account_settings['account_id'],
                live_account_settings['password'])

    def test(self, symbol: str = 'SHFE.ag2212'):
        api = TqApi(auth=self.auth)
        # quote = api.get_quote(symbol)
        tick = api.get_tick_serial(symbol)
        bar_1m = api.get_kline_serial(symbol, 60)
        bar_5m = api.get_kline_serial(symbol, 300)
        bar_30m = api.get_kline_serial(symbol, 1800)
        bar_1d = api.get_kline_serial(symbol, 86400)
        while True:
            api.wait_update()
            print("tick", tick.shape, tick.iloc[-1])
            # print("bar_1m", bar_1m.shape, bar_1m.iloc[-1])
            # print("bar_5m", bar_5m.shape, bar_5m.iloc[-1])
            # print("bar_30m", bar_30m.shape, bar_30m.iloc[-1])
            # print("bar_1d", bar_1d.shape, bar_1d.iloc[-1])
