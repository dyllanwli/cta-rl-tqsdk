from datetime import date, datetime

from API import API
# from commodity import Commodity
# from policies.load_policy import LoadPolicy
from dao.mongo import MongoDAO

from tqsdk2 import TqApi
import wandb


def main():
    tqAPI = API(account='a1')
    # cmod = Commodity()
    # symbol = cmod.get_instrument_name('egg')

    # init wandb by symbol and datetime
    # wandb.init(project="tqrl-dev",
    #            name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    mongodb = MongoDAO()
    mongodb.download_data(tqAPI.auth, ['cotton', 'methanol'], date(2016, 1, 1), date(2018, 5, 1))


if __name__ == "__main__":
    main()
