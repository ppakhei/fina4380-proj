import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
from statsmodels.tsa.api import VAR
from scipy.optimize import minimize
from liquidity_filter import LiquidityFilter
import itertools


def pairwise_diff(a):
    n = a.shape[1]
    N = n*(n-1)//2
    idx = np.concatenate([[0], np.arange(n-1, 0, -1).cumsum()])
    start, stop = idx[:-1], idx[1:]
    out = np.empty((a.shape[0], N), dtype=a.dtype)
    for j, i in enumerate(range(n-1)):
        out[:, start[j]:stop[j]] = a[:, i, None] - a[:, i+1:]
    return out


class CigSubspace:

    def __init__(self, df, n=50, adf_threshold=-2):
        self.df = df
        self.log_df = np.log(df / df.iloc[0, :])
        self.n = n
        self.adf_threshold = adf_threshold
        self.pairs = list(itertools.combinations(df.columns, 2))
        self.df_npd = self.normalized_price_distance()

        self.datetime = None
        self.npd_pairs = None
        self.beta = None
        self.df_adf = None
        self.cig_pairs = None
        self.summary = None

    def find_cig_subspace(self, datetime, year=True):
        if year:
            self.datetime = self.df.loc[str(datetime)].index[-1]
        else:
            self.datetime = datetime
        self.npd_pairs = self.df_npd.loc[self.datetime].nsmallest(self.n).index.tolist()
        self.beta, self.df_adf = self.adf()
        self.cig_pairs = self.df_adf[self.df_adf < self.adf_threshold].dropna().sort_values(by="ADF")
        self.summary = pd.concat([self.beta, self.cig_pairs], axis=1).dropna(axis=0).sort_values(by="ADF")
        self.summary.index = self.summary.index.to_flat_index()

    def normalized_price_distance(self):
        df_norm = self.df / self.df.iloc[0, :]
        df_npd = pd.DataFrame(pairwise_diff(df_norm.values)**2).cumsum()
        df_npd.index = pd.to_datetime(df_norm.index.values)
        df_npd.columns = self.pairs
        return df_npd

    def tls(self, p0, p1):

        def f(B, x):
            """Linear function y = m*x + b"""
            return B[0] + B[1] * x

        pair_y = self.log_df[p0].loc[:self.datetime]
        pair_x = self.log_df[p1].loc[:self.datetime]
        linreg = linregress(pair_x, pair_y)
        mod = Model(f)
        dat = Data(pair_x, pair_y)
        od = ODR(dat, mod, beta0=linreg[0:2])
        out = od.run()
        resid = (pair_y - out.beta[0] - out.beta[1] * pair_x) / (np.sqrt(1 + out.beta[1] ** 2))
        return out.beta[0], out.beta[1], resid

    def adf(self):
        adf_stats = []
        b0s = []
        b1s = []

        for pair in self.npd_pairs:
            b0, b1, resid = self.tls(pair[0], pair[1])
            adf_stat = adfuller(resid)[0]
            b0s.append(b0)
            b1s.append(b1)
            adf_stats.append(adf_stat)

        df_beta = pd.DataFrame([b0s, b1s], columns=self.npd_pairs)
        df_beta.index = ["B0", "B1"]
        df_adf = pd.DataFrame([adf_stats], columns=self.npd_pairs)
        df_adf.index = ["ADF"]
        return df_beta.T, df_adf.T


class MeanRevertPortfolio:

    def __init__(self, quantile=80, no_of_exceptions=2, n=50, adf_threshold=-2, target_vol=0.03, nlags=5):
        self.df = pd.read_csv('../data/spx_hist_close.csv', index_col=0, parse_dates=True)
        self.n = n
        self.adf_threshold = adf_threshold
        self.target_vol = target_vol
        self.nlags = nlags
        self.liq_filter = LiquidityFilter(close_data=self.df, quantile=quantile, no_of_exceptions=no_of_exceptions)

        self.test_df = None
        self.test_log_df = None
        self.cigs = None
        self.beta = None
        self.spread = None
        self.covs = None
        self.spread_weights = None
        self.stock_weights = None
        self.stock_weight_change = None
        self.stocks = None
        self.mrp_value = None
        self.z_stat = None

    def update_portfolio(self, datetime, new_year=True):
        self.extract_spread(datetime, new_year)
        self.calculate_portfolio(datetime)

    def extract_spread(self, datetime, new_year=True):
        if new_year:
            self.test_df = self.liq_filter.filter_uni[datetime]
            self.test_log_df = np.log(self.test_df / self.test_df.iloc[0, :])
            self.cigs = CigSubspace(self.test_df, self.n, self.adf_threshold)
            self.cigs.find_cig_subspace(datetime)
        else:
            self.cigs.find_cig_subspace(datetime, False)

        self.beta = self.cigs.summary["B1"]

        cig_pairs = self.cigs.cig_pairs
        spread = pd.DataFrame()
        for i in range(len(cig_pairs)):
            y = self.test_df[cig_pairs.index[i][0]]
            x = self.test_df[cig_pairs.index[i][1]]
            b1 = self.cigs.summary.iloc[i, 1]
            s = y - b1 * x
            s.name = cig_pairs.index[i]
            spread = pd.concat([spread, s], axis=1)
        spread.index = pd.to_datetime(spread.index.values)
        self.spread = spread

    def calculate_portfolio(self, datetime):
        self.covs = self.autocov(self.spread.loc[:str(datetime)], self.nlags)
        self.spread_weights = self.minimize_port(self.target_vol)
        stock_weights = self.decomp_spread(self.spread_weights)
        if self.stock_weights is not None:
            self.stock_weight_change = (stock_weights - self.stock_weights).dropna()
        self.stock_weights = stock_weights
        self.stocks = self.stock_weights.index.values
        self.mrp_value = (self.test_df[self.stocks] * self.stock_weights.values.T).sum(axis=1)
        self.z_stat = (self.mrp_value.loc[:str(datetime)].mean(), self.mrp_value.loc[:str(datetime)].std())

    def autocov(self, spread, nlags):
        model = VAR(spread.values)
        M = model.fit(maxlags=nlags).sample_acov(nlags=nlags)
        return M

    def variance(self, weights, M):
        return weights.T @ M @ weights

    def portmanteau(self, weights):
        M0 = self.covs[0]
        out = 0
        for M in self.covs[1:]:
            v = self.variance(weights, M) / self.variance(weights, M0)
            out = out + v * v
        return out

    def minimize_port(self, target_vol):
        n = self.covs.shape[1]
        initialguess = np.repeat(1 / n, n)
        bounds = ((-1.0, 1.0),) * n
        weights_constraint = {
            'type': 'eq',
            'fun': lambda weights: np.sum(weights)
        }
        vol_constraint = {
            'type': 'eq',
            'fun': lambda weights: self.variance(weights, self.covs[0]) - target_vol ** 2
        }
        out = minimize(self.portmanteau,
                       initialguess,
                       method="SLSQP",
                       options={'disp': False},
                       constraints=[weights_constraint, vol_constraint],
                       bounds=bounds
                       )

        return pd.DataFrame(out.x, index=self.spread.columns)

    def decomp_spread(self, spread_weights):
        long = spread_weights.copy(deep=True)
        short = -self.beta * spread_weights.T.values[0]
        long.index = [pair[0] for pair in long.index]
        short.index = [pair[1] for pair in short.index]
        out = pd.concat([long, short])
        out = out.groupby(out.index).sum()

        return out

    def remove_stock(self, stock, datetime):
        self.beta = self.beta[[stock not in pair for pair in self.spread.columns]]
        self.spread = self.spread.T[[stock not in pair for pair in self.spread.columns]].T
        self.calculate_portfolio(datetime)
