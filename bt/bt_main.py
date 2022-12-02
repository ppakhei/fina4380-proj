import backtrader as bt
import datetime

import pandas as pd

from bt_datafeed import CloseData, DataEndDate
from bt_strat import SPXStatArbitrageStrategy
from bt_analyser import PortfolioStats, StratPerformance
from bt_commission import IBCommission
from mean_reverting_portfolio import MeanRevertPortfolio

if __name__ == '__main__':
    # start_date = datetime.datetime(2010, 1, 1)
    # end_date = datetime.datetime(2015, 12, 31)
    start_date = datetime.datetime(2001, 1, 1)
    end_date = datetime.datetime(2022, 10, 31)

    params_list = [
        (90, 100, -3, 5, 3),
        (80, 200, -3, 6, 3),
        (50, 400, -4, 5, 2),
    ]
    metrics_df = pd.DataFrame()
    strat_stats = {}

    for params in params_list:
        mrp_test = MeanRevertPortfolio(quantile=params[0], n=params[1], adf_threshold=params[2])

        cerebro = bt.Cerebro()
        cerebro.addstrategy(SPXStatArbitrageStrategy, mrp=mrp_test, stat_break=params[3],
                            short_open=params[4], long_open=-params[4])
        cerebro.broker.addcommissioninfo(IBCommission())
        cerebro.addanalyzer(PortfolioStats, _name='port_stats')

        cerebro.broker.setcash(100000000)

        for stock in mrp_test.liq_filter.filter_stocks:
            dataname = f'../data/stocks/{stock}.csv'
            data = CloseData(dataname=dataname, fromdate=start_date, todate=end_date, plot=False)
            cerebro.adddata(data)
            cerebro.add_timer(when=bt.timer.SESSION_START, allow=DataEndDate(mrp_test.df[stock]),
                              strats=True, timername=stock)

        print(f'Backtest {params} start')
        results = cerebro.run()
        print(f'Backtest {params} end')

        strat = results[0]
        port_stats = strat.analyzers.port_stats.get_analysis()
        strat_performance = StratPerformance(port_stats)
        metrics = strat_performance.result()
        metrics.columns = [params]
        metrics_df = pd.concat([metrics_df, metrics], axis=1)
        strat_stats[params] = port_stats
        strat_performance.plot()
