from mean_reverting_portfolio import mrp
from liquidity_filter import liquidity_filter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    liq_filter = liquidity_filter(quantile=90, no_of_exceptions=2)
    mrp_2000 = mrp(liq_filter.filter_uni[2000])
    plt.figure(figsize=(10, 6))
    plt.plot((liq_filter.close_data.loc['2000':'2001'][mrp_2000.stocks] * mrp_2000.stock_weights.values.T).sum(axis=1))
    plt.show()
