
from datetime import datetime
from functools import reduce
from tqsdk import TqApi, TargetPosTask, TqBacktest, TqSim, BacktestFinished
from .base import BasePolicy

from tqsdk.tafunc import time_to_datetime
import wandb


class Grid(BasePolicy):

    def __init__(self):
        super().__init__()

    def run(self, api: TqApi, symbol):
        kq_quote = api.get_quote(symbol)  # 主连行情数据
        START_PRICE = kq_quote.last_price + 400
        GRID_AMOUNT = 10

        grid_region_long = [0.005] * GRID_AMOUNT
        grid_region_short = [0.005] * GRID_AMOUNT

        grid_volume_long = [i for i in range(GRID_AMOUNT + 1)]
        grid_volume_short = [i for i in range(GRID_AMOUNT + 1)]

        grid_prices_long = [reduce(
            lambda p, r: p*(1-r), grid_region_long[:i], START_PRICE) for i in range(GRID_AMOUNT + 1)]
        grid_prices_short = [reduce(
            lambda p, r: p*(1+r), grid_region_short[:i], START_PRICE) for i in range(GRID_AMOUNT + 1)]

        print("策略开始运行, 起始价位: %f, 多头每格持仓手数:%s, 多头每格的价位:%s, 空头每格的价位:%s" % (
            START_PRICE, grid_volume_long, grid_prices_long, grid_prices_short))
        

        SYMBOL = kq_quote.underlying_symbol  # 合约代码
        quote = api.get_quote(SYMBOL)  # 主连行情数据
        target_pos = TargetPosTask(api, SYMBOL)
        position = api.get_position(SYMBOL)  # 持仓信息
        account = api.get_account()  # 账户信息

        def wait_price(layer):
            """等待行情最新价变动到其他档位,则进入下一档位或回退到上一档位; 如果从下一档位回退到当前档位,则设置为当前对应的持仓手数;
                layer : 当前所在第几个档位层次; layer>0 表示多头方向, layer<0 表示空头方向
            """
            if layer > 0 or quote.last_price <= grid_prices_long[1]:  # 是多头方向
                while True:
                    api.wait_update()
                    self.update_log(api, quote, account)
                    # 如果当前档位小于最大档位,并且最新价小于等于下一个档位的价格: 则设置为下一档位对应的手数后进入下一档位层次
                    if layer < GRID_AMOUNT and quote.last_price <= grid_prices_long[layer + 1]:
                        target_pos.set_target_volume(
                            grid_volume_long[layer + 1])
                        print("最新价: %f, 进入: 多头第 %d 档" %
                              (quote.last_price, layer + 1))
                        wait_price(layer + 1)
                        # 从下一档位回退到当前档位后, 设置回当前对应的持仓手数
                        target_pos.set_target_volume(
                            grid_volume_long[layer + 1])
                    # 如果最新价大于当前档位的价格: 则回退到上一档位
                    if quote.last_price > grid_prices_long[layer]:
                        print("最新价: %f, 回退到: 多头第 %d 档" %
                              (quote.last_price, layer))
                        return
            # 是空头方向
            elif layer < 0 or quote.last_price >= grid_prices_short[1]:
                layer = -layer  # 转为正数便于计算
                while True:
                    api.wait_update()
                    self.update_log(api, quote, account)
                    # 如果当前档位小于最大档位层次,并且最新价大于等于下一个档位的价格: 则设置为下一档位对应的持仓手数后进入下一档位层次
                    if layer < GRID_AMOUNT and quote.last_price >= grid_prices_short[layer + 1]:
                        target_pos.set_target_volume(
                            -grid_volume_short[layer + 1])
                        print("最新价: %f, 进入: 空头第 %d 档" %
                              (quote.last_price, layer + 1))
                        wait_price(-(layer + 1))
                        # 从下一档位回退到当前档位后, 设置回当前对应的持仓手数
                        target_pos.set_target_volume(
                            -grid_volume_short[layer + 1])
                    # 如果最新价小于当前档位的价格: 则回退到上一档位
                    if quote.last_price < grid_prices_short[layer]:
                        print("最新价: %f, 回退到: 空头第 %d 档" %
                              (quote.last_price, layer))
                        return
        while True:
            api.wait_update()
            wait_price(0)
            target_pos.set_target_volume(0)

    def update_log(self, api, quote, account):
        if api.is_changing(quote, "datetime"):
            now = time_to_datetime(quote.datetime)
            if self.end_of_day(now):
                print("当前时间: %s" % now)
                self.wandb_log(account, now)
