import datetime

from API import API
from commodity import Commodity
from policies.load_policy import LoadPolicy
from tqsdk.tafunc import time_to_datetime

from tqsdk import TqApi
import wandb

def main():
    tqAPI = API(account='a3')
    cmod = Commodity()
    lp = LoadPolicy()
    symbol = cmod.get_kq_name('egg')

    # init wandb by symbol and datetime
    wandb.init(project="tqrl-dev", name=symbol + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    lp.load_policy('grid').backtest(tqAPI.auth, symbol, start_dt=datetime.date(
        2018, 9, 10), end_dt=datetime.date(2018, 11, 16))


if __name__ == "__main__":
    main()
