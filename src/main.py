from datetime import date, datetime
from API import API

from policies.trainer import RLTrainer

from tqsdk import TqApi
import wandb

def main():
    # tqAPI = API(account='a4')
    # tqAPI.test()
    # init wandb by symbol and datetime
    # wandb.init(project="tqrl-dev",
            #    name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    rl_trainer = RLTrainer()
    rl_trainer.train()

if __name__ == "__main__":
    main()
