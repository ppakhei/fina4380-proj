import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PortfolioStats(bt.analyzers.Analyzer):
    def __init__(self):
        super(PortfolioStats, self).__init__()
        self.stats = {}
        self.stat = None
        self.high_water_mark = 0
        self.commission = 0

    def notify_order(self, order):
        if order.status == order.Completed:
            self.commission += order.executed.comm

    def notify_cashvalue(self, cash, value):
        if value > self.high_water_mark:
            self.high_water_mark = value
        self.stat = (
            self.strategy.datetime.datetime().strftime("%Y-%m-%d"),
            cash,
            value,
            self.commission,
            (value - self.high_water_mark) / self.high_water_mark,
        )
        self.stats[len(self)] = self.stat

    def get_analysis(self):
        stats_df = pd.DataFrame(self.stats).T
        stats_df.columns = ['Date', 'Cash', 'Portfolio Value', 'Commission', 'Drawdown']
        stats_df = stats_df.set_index('Date').astype(float)
        stats_df.index = pd.to_datetime(stats_df.index)
        return stats_df


class StratPerformance:
    def __init__(self, stats_df):
        self.port_val = stats_df['Portfolio Value']
        self.drawdown = stats_df['Drawdown']
        self.total_comm = stats_df['Commission'].iloc[-1]
        self.port_ret = np.log(self.port_val).diff().dropna()

        self.cum_ret = self.port_val.iloc[-1] / self.port_val.iloc[0] - 1
        self.annual_ret = (self.port_val.iloc[-1] / self.port_val.iloc[0]) ** (252 / len(self.port_val)) - 1
        self.annual_vol = self.port_ret.std() * 16
        self.annual_down_vol = self.port_ret[self.port_ret < 0].std() * 16
        self.max_drawdown = -min(self.drawdown)

        self.sharpe = self.annual_ret / self.annual_vol
        self.sortino = self.annual_ret / self.annual_down_vol
        self.calmar = self.annual_ret / self.max_drawdown
        self.var_95 = -np.percentile(self.port_ret, 5) * 16
        self.es_95 = -(self.port_ret[self.port_ret < np.percentile(self.port_ret, 5)]).mean() * 16

    def result(self):
        result_df = pd.DataFrame(
            data=[self.cum_ret, self.annual_ret, self.annual_vol, self.max_drawdown,
                  self.sharpe, self.sortino, self.calmar, self.var_95, self.es_95],
            index=['Cumulative Return', 'Annualised Geometric Return', 'Annualised Volatility', 'Maximum Drawdown',
                   'Annualised Sharpe Ratio', 'Annualised Sortino Ratio', 'Annualised Calmar Ratio',
                   '95% 1 Year VaR', '95% 1 Year ES'],
            columns=['Metrics']
        )
        result_df.iloc[[0, 4, 5, 6], 0] = result_df.iloc[[0, 4, 5, 6], 0].map('{:,.4f}'.format)
        result_df.iloc[[1, 2, 3, -1, -2], 0] = result_df.iloc[[1, 2, 3, -1, -2], 0].map('{:,.4%}'.format)
        return result_df

    def plot(self):
        fig, axs = plt.subplots(2, figsize=(12, 8))
        axs[0].plot(self.port_val)
        axs[0].grid()
        axs[0].set_title('Portfolio Value')
        axs[1].plot(self.drawdown)
        axs[1].grid()
        axs[1].set_title('Portfolio Drawdown')
        plt.show()

