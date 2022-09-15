import datetime

from API import API
from commodity import Commodity
from policies.policy import Policy

from tqsdk import TqApi, TqBacktest


def main():
    tqAPI = API(account='a1')
    api = tqAPI.api
    cmod = Commodity()
    symbol = cmod.getCommodityConfig('铁矿石') + "2209"
    backtest = TqBacktest(
        start_dt=datetime.date(2022, 3, 1), end_dt=datetime.date(2022, 7, 1))
    policy = Policy()

    policy.load_policy('vwap').backtest(api._auth, symbol, backtest)


if __name__ == "__main__":
    main()
