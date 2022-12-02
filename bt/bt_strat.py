import numpy as np
import backtrader as bt


class SPXStatArbitrageStrategy(bt.Strategy):
    params = (
        ('mrp', None),
        ('short_open', 2),
        ('long_open', -2),
        ('short_close', 0),
        ('long_close', 0),
        ('stat_break', 6),
        ('init_margin', 1.5),
        ('mtn_margin', 1.25),
    )

    def __init__(self):
        self.mrp = self.p.mrp
        self.cur_year = None
        self.mrp_val = None
        self.z_score = None
        self.port_size = None
        self.short = False
        self.long = False

    def log(self, txt):
        print(f'{self.datetime.date(0)}: {txt}')

    def next(self):
        # if np.isnan(self.broker.getvalue()):
        # self.log('ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR')

        # self.log(str(f'Portfolio value {self.broker.getvalue():,.2f}, cash value {self.broker.getcash():,.2f}'))

        if self.short or self.long:
            short_pos = 0
            for pos in [self.broker.getposition(position) for position in self.broker.positions]:
                if pos.size < 0:
                    short_pos += pos.size * pos.adjbase
            margin_level = -self.broker.getcash() / short_pos
            if margin_level < self.p.mtn_margin:
                # self.log('Margin call on portfolio')
                self.margin_call_portfolio()

        if self.cur_year != self.datetime.date(0).year:
            if self.cur_year is not None:
                self.short = False
                self.long = False
                # self.log('Close portfolio position')
                self.close_portfolio()
            self.cur_year = self.datetime.date(0).year
            # self.log('Generate annual portfolio')
            self.mrp.update_portfolio(self.cur_year - 1)

        self.calculate_portfolio_z_score()

        if np.abs(self.z_score) > self.p.stat_break:
            self.short = False
            self.long = False
            # self.log('Close portfolio position')
            self.close_portfolio()
            # self.log('Generate stationary portfolio')
            self.mrp.update_portfolio(self.datetime.date(0).strftime('%Y-%m-%d'), False)
            self.calculate_portfolio_z_score()

        if self.z_score < self.p.short_close and self.short:
            self.short = False
            # self.log('Close short portfolio position')
            self.close_portfolio()
        elif self.z_score > self.p.long_close and self.long:
            self.long = False
            # self.log('Close long portfolio position')
            self.close_portfolio()

        if self.z_score > self.p.short_open and not self.short:
            self.short = True
            # self.log('Short portfolio position')
            self.create_portfolio_position('SHORT')
        elif self.z_score < self.p.long_open and not self.long:
            self.long = True
            # self.log('Long portfolio position')
            self.create_portfolio_position('LONG')

    def notify_timer(self, timer, when, *args, **kwargs):
        end_stock = kwargs.get('timername')
        # self.log(str(f'{end_stock} END'))
        if end_stock in self.mrp.stocks:
            # self.log(str(f'Remove {end_stock} from portfolio'))
            # self.log(f'Close {end_stock} position')
            self.close(self.dnames[end_stock])
            if self.datetime.date(0).month != 12:
                # self.log('Rebalance portfolio')
                self.mrp.remove_stock(end_stock, self.datetime.date(0).strftime('%Y-%m-%d'))
                self.rebalance_portfolio()

    def calculate_portfolio_z_score(self):
        self.mrp_val = [0, 0]
        for stock in self.mrp.stocks:
            stock_val = self.dnames[stock].close[0] * self.mrp.stock_weights.loc[stock][0]

            if stock_val > 0:
                self.mrp_val[0] += stock_val
            else:
                self.mrp_val[1] += stock_val
        # self.log(str(f'MRP value {sum(self.mrp_val) * 1000000:.2f}'))
        self.z_score = (sum(self.mrp_val) - self.mrp.z_stat[0]) / self.mrp.z_stat[1]

    def create_portfolio_position(self, direction):
        if direction == 'LONG':
            self.port_size = self.broker.getcash() / (self.mrp_val[0] - (self.p.init_margin - 1) * self.mrp_val[1])
        else:
            self.port_size = self.broker.getcash() / ((self.p.init_margin - 1) * self.mrp_val[0] - self.mrp_val[1])
        for stock in self.mrp.stocks:
            if direction == 'LONG':
                size = int(self.mrp.stock_weights.loc[stock][0] * self.port_size)
            else:
                size = -int(self.mrp.stock_weights.loc[stock][0] * self.port_size)
            if size > 0:
                self.buy(self.dnames[stock], size)
            else:
                self.sell(self.dnames[stock], np.abs(size))

    def close_portfolio(self):
        for stock in self.mrp.stocks:
            self.close(self.dnames[stock])

    def rebalance_portfolio(self):
        for stock in self.mrp.stocks:
            size = int(self.mrp.stock_weight_change.loc[stock][0] * self.port_size)
            if size > 0:
                self.buy(self.dnames[stock], size)
            else:
                self.sell(self.dnames[stock], np.abs(size))

    def margin_call_portfolio(self):
        if self.long:
            new_port_size = self.broker.getcash() / ((self.p.init_margin - 1) * self.mrp_val[0] - self.mrp_val[1])
        else:
            new_port_size = self.broker.getcash() / ((self.p.init_margin - 1) * self.mrp_val[0] - self.mrp_val[1])
        size_change = self.port_size - new_port_size
        self.port_size = new_port_size
        for stock in self.mrp.stocks:
            if self.long:
                size = -int(self.mrp.stock_weights.loc[stock][0] * size_change)
            else:
                size = int(self.mrp.stock_weights.loc[stock][0] * size_change)
            if size > 0:
                self.buy(self.dnames[stock], size)
            else:
                self.sell(self.dnames[stock], np.abs(size))

