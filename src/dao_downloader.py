from datetime import date, datetime
from API import API
# from commodity import Commodity
# from policies.load_policy import LoadPolicy
from dao.mongo import MongoDAO

# from policies.trainer import RLTrainer

from tqsdk import TqApi
import wandb

def main():
    tqAPI = API(account='a1')
    # tqAPI.test()
    # cmod = Commodity()
    # symbol = cmod.get_instrument_name('egg')
    dao = MongoDAO()
    # dao.download_data(tqAPI.auth, ['cotton', 'methanol'], date(2016, 10, 25), date(2022, 2, 1))
    # dao.download_data(tqAPI.auth, ['rebar'], date(2016, 10, 25), date(2022, 2, 1))
    intervals = {'1s', '1d'}
    dao.download_data(tqAPI.auth, ['soybean'], date(2016, 11, 2), date(2022, 9, 1), intervals)
    # init wandb by symbol and datetime
    # wandb.init(project="tqrl-dev",
            #    name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # rl_trainer = RLTrainer(auth=tqAPI.auth)
    # rl_trainer.train()

if __name__ == "__main__":
    main()
