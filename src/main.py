import datetime

from API import API
from commodity import Commodity
from policies.policy import Policy

from tqsdk import TqApi


def main():
    tqAPI = API(account='a1')
    api = tqAPI.api
    cmod = Commodity()
    symbol = cmod.getCommodityConfig('铁矿石') + "2209"
    policy = Policy()

    policy.load_policy('r_breaker').backtest(api._auth, symbol, start_dt=datetime.date(
        2022, 3, 1), end_dt=datetime.date(2022, 7, 1))


if __name__ == "__main__":
    main()
