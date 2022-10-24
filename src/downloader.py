from datetime import date, datetime
from API import API
# from commodity import Commodity
# from policies.load_policy import LoadPolicy
from dao.mongo import MongoDAO

# from policies.trainer import RLTrainer

from tqsdk import TqApi
import wandb

def main():
    tqAPI = API(account='a2')
    dao = MongoDAO()
    intervals = {'1s', '5s', '1m', '1d'}
    symbol_list = ['soybean_oil']
    dao.download_data(tqAPI.auth, symbol_list, date(2020, 9, 3), date(2022, 9, 1), intervals)
    # intervals =  {'1s', '5s', '1m', '1d'}
    # symbol_list = ['methanol']
    # dao.download_data(tqAPI.auth, symbol_list, date(2016, 7, 4), date(2022, 9, 1), intervals)
    print(symbol_list, intervals, "downloaded")
if __name__ == "__main__":
    main()


