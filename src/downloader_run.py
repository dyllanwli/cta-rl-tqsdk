from datetime import date, datetime
from utils.api import API
from dao.mongo import MongoDAO

from tqsdk import TqApi
import wandb

def main():
    tqAPI = API(account='a2')
    dao = MongoDAO()
    # intervals = {'1s', '5s', '1m', '1d'}
    # symbol_list = ['soybean_oil']
    # dao.download_data(tqAPI.auth, symbol_list, date(2020, 9, 3), date(2022, 9, 1), intervals)
    intervals =  {'1s', '5s', '1m', '1d'}
    symbol_list = ['methanol']
    dao.download_data(tqAPI.auth, symbol_list, date(2021, 10, 6), date(2022, 9, 1), intervals)
    print(symbol_list, intervals, "downloaded")
if __name__ == "__main__":
    main()


