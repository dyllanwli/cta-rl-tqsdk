from datetime import date, datetime
from API import API
# from commodity import Commodity
# from policies.load_policy import LoadPolicy
from dao.mongo import MongoDAO

# from policies.trainer import RLTrainer

from tqsdk import TqApi
import wandb

def main():
    tqAPI = API(account='a3')
    dao = MongoDAO()
    # dao.download_data(tqAPI.auth, ['cotton', 'methanol'], date(2016, 10, 25), date(2022, 2, 1))
    # dao.download_data(tqAPI.auth, ['rebar'], date(2016, 10, 25), date(2022, 2, 1))
    intervals = {'1s', '5s', '1m', '1d'}
    symbol_list = ['soybean_oil']
    dao.download_data(tqAPI.auth, symbol_list, date(2016, 1, 1), date(2022, 9, 1), intervals)
    print(symbol_list, intervals, "downloaded")
if __name__ == "__main__":
    main()


