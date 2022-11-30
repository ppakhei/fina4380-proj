import pandas as pd
import numpy as np


class liquidity_filter:

    def __init__(self, quantile=80, no_of_exceptions=2):
        self.close_data = pd.read_csv('../data/spx_hist_close.csv', index_col=0, parse_dates=True)
        self.vol_data = pd.read_csv('../data/spx_hist_volume.csv', index_col=0, parse_dates=True)
        self.vol_thres = self.vol_data.apply(np.nanpercentile, axis=1, q=quantile)
        self.vol_filter = self.vol_data.apply(lambda x: np.where(x < self.vol_thres, True, False)).resample('Y').sum()
        self.vol_filter = self.vol_filter <= no_of_exceptions
        self.filter_uni, self.filter_stocks = self.get_filter_uni()

    def get_filter_uni(self):
        train_period = dict()

        for i in range(self.vol_filter.shape[1]):
            train_year = self.vol_filter.iloc[:, i][self.vol_filter.iloc[:, i]].index.year
            train_filter = self.close_data.index.year.isin(train_year)
            train_data = self.close_data.iloc[:, i].loc[train_filter].dropna()
            if not train_data.empty:
                train_period[train_data.name] = [train_data.index.year.unique().values]

        train_period = pd.DataFrame.from_dict(train_period)
        filter_uni = {i: self.close_data[train_period.T[train_period.apply(
            lambda x: i in x[0])].index.values].loc[str(i)].dropna(axis=1) for i in range(2000, 2022)}

        filter_stocks = []
        for filter_df in filter_uni.values():
            filter_stocks.extend(filter_df.columns.values.tolist())
        filter_stocks = set(filter_stocks)

        return filter_uni, filter_stocks
