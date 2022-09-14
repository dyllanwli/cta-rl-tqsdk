import datetime

from API import API
from commodity import Commodity
from policies.policy import Policy

from tqsdk import TqApi, TqBacktest


def main():
    tqAPI = API()
    api = tqAPI.api
    auth = tqAPI.auth
    cmod = Commodity()
    symbol = cmod.getCommodityConfig('铁矿石') + "2301"
    backtest = TqBacktest(start_dt=datetime.date(
        2018, 7, 2), end_dt=datetime.date(2018, 9, 1))

    quote = api.get_quote(symbol)
    print(quote)
    policy = Policy()
    api.close()


    # policy.load_policy('random_forest')(auth, symbol, backtest)
    # policy.load_policy('vwap')(api, symbol)

    # quote = api.get_quote(symbol)
    # while True:
    #     api.wait_update()
    #     print(quote)


if __name__ == "__main__":
    main()
