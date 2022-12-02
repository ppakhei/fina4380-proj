import backtrader.feeds as btfeeds


class CloseData(btfeeds.GenericCSVData):
    params = (
        ('dtformat', ('%Y-%m-%d')),
        ('datetime', 0),
        ('time', -1),
        ('high', -1),
        ('low', -1),
        ('open', 1),
        ('close', 1),
        ('volume', -1),
        ('openinterest', -1)
    )


class DataEndDate:
    def __init__(self, stock_data):
        self.stock_enddate = ((1*stock_data.isna()).diff() == 1).shift(-2).fillna(False)

    def __call__(self, d):
        return self.stock_enddate[d.strftime('%Y-%m-%d')]