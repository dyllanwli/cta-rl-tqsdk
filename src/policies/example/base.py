from datetime import datetime
from tqsdk import TqApi, TargetPosTask, TqBacktest, TqSim, BacktestFinished
from contextlib import closing


class BasePolicy:
    def __init__(self):
        self.day_kline = 24 * 60 * 60 # day kline

    def run(self, api, symbol):
        pass

    def strptime(self, quote_time):
        return datetime.strptime(quote_time, "%Y-%m-%d %H:%M:%S.%f")

    def backtest(self, auth, symbol, start_dt, end_dt, debug=False):
        print("开始回测")

        acc = TqSim(init_balance=500000)
        backtest = TqBacktest(start_dt, end_dt)
        if debug:
            api = TqApi(acc, backtest=backtest, auth=auth, debug="./debugs/%s.log" %
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            api = TqApi(acc, backtest=backtest, auth=auth)
        with closing(api):
            try:
                self.run(api, symbol)

            except BacktestFinished as e:
                print("回测结束")
                account = api.get_account()
                print(acc.trade_log)  # 回测的详细信息
                print(acc.tqsdk_stat)  
                print(account.balance)  # 回测结束时的账户权益
                # init_balance 起始资金
                # balance 结束资金
                # max_drawdown 最大回撤
                # profit_loss_ratio 盈亏额比例
                # winning_rate 胜率
                # ror 收益率
                # annual_yield 年化收益率
                # sharpe_ratio 年化夏普率
                # tqsdk_punchline 天勤点评
            except IndexError as e:
                print("回测结束(IndexError)")
                account = api.get_account()
                print(acc.trade_log)
                print(acc.tqsdk_stat)  
