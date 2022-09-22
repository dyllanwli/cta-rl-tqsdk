import datetime

from API import API
from commodity import Commodity
from policies.load_policy import LoadPolicy

from tqsdk import TqApi


def main():
    tqAPI = API(account='a1')
    api = tqAPI.api
    cmod = Commodity()
    lp = LoadPolicy()
    symbol = cmod.get_kq_name('iron_orb')

    lp.load_policy('random_forest').backtest(api._auth, symbol, start_dt=datetime.date(
        2021, 7, 1), end_dt=datetime.date(2022, 7, 1))

    api.close()


if __name__ == "__main__":
    main()
