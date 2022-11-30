import backtrader as bt
import datetime
from bt_datafeed import CloseData
from bt_strat import SPXStatArbitrageStrategy
from liquidity_filter import liquidity_filter

if __name__ == '__main__':
    liq_filter = liquidity_filter()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SPXStatArbitrageStrategy)

    cerebro.broker.setcash(100000000)

    for stock_data in [f'../data/stocks/{stock}.csv' for stock in liq_filter.filter_stocks]:
        data = CloseData(dataname=stock_data,
                         fromdate=datetime.datetime(2001, 1, 1),
                         todate=datetime.datetime(2021, 12, 31))
        cerebro.adddata(data)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
