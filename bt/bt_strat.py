import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt
from mean_reverting_portfolio import mrp
from liquidity_filter import liquidity_filter


class SPXStatArbitrageStrategy(bt.Strategy):
    params = (
        ('quantile', 80),
        ('no_of_exceptions', 2)
    )

    def __int__(self):
        self.liq_filter = liquidity_filter(quantile=self.params.quantile, no_of_exceptions=self.params.no_of_exceptions)

    def next(self):
        self.log()

    def log(self):
        print(f'{self.datas[0].datetime.date(0)}')
