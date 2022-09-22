# !/usr/bin/env python
#  -*- coding: utf-8 -*-
from datetime import datetime
from tqsdk import TqApi, TargetPosTask, TqBacktest, TqSim, BacktestFinished
from .base import BasePolicy




class MACD_Oscillator(BasePolicy):

    def __init__(self):
        super(MACD_Oscillator).__init__()

    def run(self, api, symbol):
        """
        MACD Oscillator

        signal generation:
        when the short moving average is larger than long moving average, we long and hold
        when the short moving average is smaller than long moving average, we clear positions
        the logic behind this is that the momentum has more impact on short moving average
        we can subtract short moving average from long moving average
        the difference between is sometimes positive, it sometimes becomes negative
        thats why it is named as moving average converge/diverge oscillator
        """
        SYMBOL = symbol
        CLOSE_HOUR, CLOSE_MINUTE = 1, 50

        