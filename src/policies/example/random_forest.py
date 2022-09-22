#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import pandas as pd
import datetime
from contextlib import closing
from tqsdk import TqApi, TqBacktest, BacktestFinished, TargetPosTask
from tqsdk.tafunc import sma, ema2, trma
from sklearn.ensemble import RandomForestClassifier

from .base import BasePolicy

pd.set_option('display.max_rows', None)  # 设置Pandas显示的行数
pd.set_option('display.width', None)  # 设置Pandas显示的宽度


class RandomForest(BasePolicy):

    def __init__(self) -> None:
        super(RandomForest, self).__init__()
        self.current_pos = 0

    def update_quote(self, api, kq_quote):
        self.underlying_symbol = kq_quote.underlying_symbol
        quote = api.get_quote(self.underlying_symbol)
        return quote

    def change_target_pos(self, target_pos: TargetPosTask):
        # move near contract to deferred contract if underlying symbol is changed
        target_pos.set_target_volume(0)
        target_pos = TargetPosTask(self.api, self.underlying_symbol)
        target_pos.set_target_volume(self.current_pos)
        return target_pos

    def get_prediction_data(self, klines, n):
        """获取用于随机森林的n个输入数据(n为数据长度): n天中每天的特征参数及其涨跌情况"""
        close_prices = klines.close[- 30 - n:]
        # 获取本交易日及以前的收盘价(此时在预定的收盘时间: 认为本交易日已收盘)
        # 计算所需指标
        sma_data = sma(close_prices, 30, 0.02)[-n:]  # SMA指标, 函数默认时间周期参数:30
        wma_data = ema2(close_prices, 30)[-n:]  # WMA指标
        mom_data = trma(close_prices, 30)[-n:]  # MOM指标
        x_all = list(zip(sma_data, wma_data, mom_data))  # 样本特征组
        y_all = list(klines.close.iloc[i] >= klines.close.iloc[i - 1]
                     for i in list(reversed(range(-1, -n - 1, -1))))  # 样本标签组
        # x_all:            大前天指标 前天指标 昨天指标 (今天指标)
        # y_all:   (大前天)    前天     昨天    今天      -明天-
        # 准备算法需要用到的数据
        x_train = x_all[: -1]  # 训练数据: 特征
        x_predict = x_all[-1]  # 预测数据(用本交易日的指标预测下一交易日的涨跌)
        # 训练数据: 标签 (去掉第一个数据后让其与指标隔一位对齐(例如: 昨天的特征 -> 对应预测今天的涨跌标签))
        y_train = y_all[1:]

        return x_train, y_train, x_predict

    def run(self, api: TqApi, symbol):
        """
        随机森林策略
        input:
            auth:       账户信息
            symbol:     合约代码
            backtest:   回测对象
        """
        print("start random_forest")
        self.api = api

        close_hour, close_minute = 14, 50  # 预定收盘时间(因为真实收盘后无法进行交易, 所以提前设定收盘时间)
        predictions = []  # 用于记录每次的预测结果(在每个交易日收盘时用收盘数据预测下一交易日的涨跌,并记录在此列表里)

        kq_quote = api.get_quote(symbol)
        quote = self.update_quote(api, kq_quote)
        klines = api.get_kline_serial(symbol, duration_seconds=self.day_kline)
        target_pos = TargetPosTask(api, symbol)
        while True:
            # 等到达下一个交易日
            while not api.is_changing(klines.iloc[-1], "datetime"):
                api.wait_update()
            while True:
                api.wait_update()
                if api.is_changing(kq_quote, "underlying_symbol"):
                    quote = self.update_quote(api, kq_quote)
                    target_pos = self.change_target_pos(target_pos)
                # 在收盘后预测下一交易日的涨跌情况
                if api.is_changing(quote, "datetime"):
                    now = datetime.datetime.strptime(
                        quote.datetime, "%Y-%m-%d %H:%M:%S.%f")  # 当前quote的时间
                    # 判断是否到达预定收盘时间: 如果到达 则认为本交易日收盘, 此时预测下一交易日的涨跌情况, 并调整为对应仓位
                    if now.hour == close_hour and now.minute >= close_minute:
                        # 1- 获取数据
                        x_train, y_train, x_predict = self.get_prediction_data(
                            klines, 90)

                        # 2- 利用机器学习算法预测下一个交易日的涨跌情况
                        # n_estimators 参数: 选择森林里（决策）树的数目; bootstrap 参数: 选择建立决策树时，是否使用有放回抽样
                        clf = RandomForestClassifier(
                            n_estimators=30, bootstrap=True)
                        clf.fit(x_train, y_train)  # 传入训练数据, 进行参数训练
                        # 传入测试数据进行预测, 得到预测的结果
                        predictions.append(bool(clf.predict([x_predict])))

                        # 3- 进行交易
                        if predictions[-1] == True:  # 如果预测结果为涨: 买入
                            print(quote.datetime, "预测下一交易日为 涨")
                            self.current_pos = 10
                            target_pos.set_target_volume(self.current_pos)
                        else:  # 如果预测结果为跌: 卖出
                            print(quote.datetime, "预测下一交易日为 跌")
                            self.current_pos = -10
                            target_pos.set_target_volume(self.current_pos)
                        break
