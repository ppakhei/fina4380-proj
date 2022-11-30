import datetime
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
