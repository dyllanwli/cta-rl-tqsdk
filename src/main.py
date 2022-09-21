import datetime

from API import API
from commodity import Commodity
from policies.policy import Policy

from tqsdk import TqApi


def main():
    tqAPI = API(account='a1')
    api = tqAPI.api
    cmod = Commodity()
    policy = Policy()
    symbol = cmod.get_kq_name('iron_orb')
    quote = api.get_quote(symbol)
    underlying_symbol = quote.underlying_symbol
    while True:
        print(datetime.datetime.now(), quote)
    # policy.load_policy('r_breaker_overnight').backtest(api._auth, symbol, start_dt=datetime.date(
    #     2022, 3, 1), end_dt=datetime.date(2022, 7, 1))

    api.close()


if __name__ == "__main__":
    main()
