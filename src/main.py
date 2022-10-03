from datetime import date, datetime
from API import API
# from commodity import Commodity
# from policies.load_policy import LoadPolicy
# from dao.mongo import MongoDAO

from policies.trainer import RLTrainer

from tqsdk import TqApi
import wandb

def main():
    tqAPI = API(account='a2')
    # tqAPI.test()
    # cmod = Commodity()
    # symbol = cmod.get_instrument_name('egg')

    # init wandb by symbol and datetime
    # wandb.init(project="tqrl-dev",
    #            name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    rl_trainer = RLTrainer(auth=tqAPI.auth)
    rl_trainer.train()

if __name__ == "__main__":
    main()
