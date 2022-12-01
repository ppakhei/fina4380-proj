import numpy as np
import backtrader as bt


class SPXStatArbitrageStrategy(bt.Strategy):
    params = (
        ('mrp', None),
    )

    def __init__(self):
        self.mrp = self.p.mrp
        self.cur_year = None
        self.z_score = None
        self.short = False
        self.long = False

    def log(self, txt):
        print(f'{self.datetime.date(0)}: {txt}')

    def next(self):
        if np.isnan(self.broker.getvalue()):
            self.log('ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR')

        if self.cur_year != self.datetime.date(0).year:
            self.cur_year = self.datetime.date(0).year
            self.log('Generate annual portfolio')
            self.mrp.update_portfolio(self.cur_year - 1)
            self.short = False
            self.long = False
            self.log('Close portfolio position')
            self.close_portfolio()

        self.calculate_portfolio_z_score()

        if np.abs(self.z_score) > 6:
            self.short = False
            self.long = False
            self.log('Close portfolio position')
            self.close_portfolio()
            self.log('Generate stationary portfolio')
            self.mrp.update_portfolio(self.datetime.date(0).strftime('%Y-%m-%d'), False)
            self.calculate_portfolio_z_score()
            self.log(str(f'Portfolio z_score {self.z_score:.2f}'))

        if self.z_score < 0 and self.short:
            self.short = False
            self.log('Close short portfolio position')
            self.close_portfolio()
        elif self.z_score > 0 and self.long:
            self.long = False
            self.log('Close long portfolio position')
            self.close_portfolio()

        if self.z_score > 2 and not self.short:
            self.short = True
            self.log('Short portfolio position')
            self.create_portfolio_position('SHORT')
        elif self.z_score < -2 and not self.long:
            self.long = True
            self.log('Long portfolio position')
            self.create_portfolio_position('LONG')

    def notify_timer(self, timer, when, *args, **kwargs):
        end_stock = kwargs.get('timername')
        self.log(str(f'{end_stock} END'))
        if end_stock in self.mrp.stocks:
            self.log(str(f'Remove {end_stock} from portfolio'))
            if self.datetime.date(0).month != 12:
                self.log('Rebalance portfolio')
                self.mrp.remove_stock(end_stock, self.datetime.date(0).strftime('%Y-%m-%d'))
            else:
                self.log(f'Close {end_stock} position')
                self.close(self.dnames[end_stock])

    def calculate_portfolio_z_score(self):
        port_val = 0
        for stock in self.mrp.stocks:
            port_val += self.dnames[stock].close[0] * self.mrp.stock_weights.loc[stock][0]
        self.z_score = (port_val - self.mrp.z_stat[0]) / self.mrp.z_stat[1]

    def create_portfolio_position(self, direction):
        for stock in self.mrp.stocks:
            if direction == 'LONG':
                size = int(self.mrp.stock_weights.loc[stock][0] * 100000)
            else:
                size = -int(self.mrp.stock_weights.loc[stock][0] * 100000)
            if size > 0:
                self.buy(self.dnames[stock], size)
            else:
                self.sell(self.dnames[stock], size)

    def close_portfolio(self):
        for stock in self.mrp.stocks:
            self.close(self.dnames[stock])
        self.log(str(f'Portfolio value {self.broker.getvalue()}'))

    def rebalance_portfolio(self):
        pass
