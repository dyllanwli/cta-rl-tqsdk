from datetime import datetime
from tqsdk import TqApi, TargetPosTask, TqBacktest, TqSim, BacktestFinished
from tqsdk.objs import Account
from contextlib import closing

import wandb


class BasePolicy:
    def __init__(self):
        self.day_kline = 24 * 60 * 60  # day kline

    def run(self, api, symbol):
        pass

    def end_of_day(self, now):
        return now.hour == 1 and now.minute == 59 and now.second == 59 and now.microsecond >= 999999

    def wandb_log(self, account: Account):
        wandb.log({
            # "currency": account.currency,
            "pre_balance": account.pre_balance,
            "static_balance": account.static_balance,
            "balance": account.balance,
            "available": account.available,
            "float_profit": account.float_profit,
            "position_profit": account.position_profit,
            "close_profit": account.close_profit,
            "frozen_margin": account.frozen_margin,
            "margin": account.margin,
            "frozen_commission": account.frozen_commission,
            "commission": account.commission,
            "frozen_premium": account.frozen_premium,
            "premium": account.premium,
            "risk_ratio": account.risk_ratio,
            "market_value": account.market_value,
        })

    def backtest(self, auth, symbol, start_dt, end_dt, debug=False):
        print("开始回测")

        acc = TqSim(init_balance=500000)
        backtest = TqBacktest(start_dt, end_dt)
        if debug:
            api = TqApi(acc, backtest=backtest, auth=auth, debug="./debugs/%s.log" %
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), web_gui=True)
        else:
            api = TqApi(acc, backtest=backtest, auth=auth)
        with closing(api):
            try:
                self.run(api, symbol)

            except BacktestFinished as e:
                print("回测结束")
                self.backtest_log(api, acc)
            except IndexError as e:
                print("回测结束(IndexError)")
                self.backtest_log(api, acc)

    def backtest_log(self, api: TqApi, acc: TqSim):
        account = api.get_account()
        # print("回测的详细信息", acc.trade_log)
        # print("回测状态", acc.tqsdk_stat)
        print("回测结束时的账户权益", account.balance)
