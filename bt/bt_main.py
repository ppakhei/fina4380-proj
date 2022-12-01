import backtrader as bt
import datetime
import quantstats
from bt_datafeed import CloseData
from bt_strat import SPXStatArbitrageStrategy
from liquidity_filter import data_enddate
from mean_reverting_portfolio import mrp

if __name__ == '__main__':
    start_date = datetime.datetime(2001, 1, 3)
    end_date = datetime.datetime(2022, 10, 31)
    mrp_test = mrp(quantile=90, n=200)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SPXStatArbitrageStrategy, mrp=mrp_test)
    # cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

    cerebro.broker.setcash(100000000)

    for stock in mrp_test.liq_filter.filter_stocks:
        dataname = f'../data/stocks/{stock}.csv'
        data = CloseData(dataname=dataname, fromdate=start_date, todate=end_date, plot=False)
        cerebro.adddata(data)
        cerebro.add_timer(when=bt.timer.SESSION_START, allow=data_enddate(mrp_test.df[stock]),
                          strats=True, timername=stock)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()
    # strat = results[0]

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # portfolio_stats = strat.analyzers.getbyname('PyFolio')
    # returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    # returns.index = returns.index.tz_convert(None)
    # quantstats.reports.html(returns, output='stats.html', title='SPX Statistical Arbitrage')
