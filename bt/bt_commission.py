import backtrader as bt
import numpy as np


class IBCommission(bt.CommInfoBase):
    params = (
        ('per_share', 0.005),
        ('min_per_order', 1.0),
        ('max_per_order_abs_pct', 0.01),
        ('interest', 0.01),
    )

    def _getcommission(self, size, price, pseudoexec):
        commission = np.abs(size) * self.p.per_share
        order_price = price * np.abs(size)
        commission_as_percentage_of_order_price = commission / order_price

        if commission < self.p.min_per_order:
            commission = self.p.min_per_order
        elif commission_as_percentage_of_order_price > self.p.max_per_order_abs_pct:
            commission = order_price * self.p.max_per_order_abs_pct
        return commission
