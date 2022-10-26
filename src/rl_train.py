
from policies.trainer import RLTrainer


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
