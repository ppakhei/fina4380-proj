import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
from statsmodels.tsa.api import VAR
from scipy.optimize import minimize
import itertools


class cig_subspace:

    def __init__(self, df, n=50, adf_threshold=-2):
        self.df = df
        self.log_df = np.log(df / df.iloc[0, :])
        self.n = n
        self.adf_threshold = adf_threshold
        self.pairs = list(itertools.combinations(df.columns, 2))

        self.df_npd = self.normalized_price_distance()
        self.npd_pairs = self.df_npd.nsmallest(self.n, "NPD").index.tolist()

        self.beta, self.df_adf = self.adf()
        self.cig_pairs = self.df_adf[self.df_adf < self.adf_threshold].dropna().sort_values(by="ADF")
        self.summary = pd.concat([self.beta, self.cig_pairs], axis=1).dropna(axis=0).sort_values(by="ADF")

    def normalized_price_distance(self):
        df_norm = self.df / self.df.iloc[0, :]
        NPDs = []

        for pair in self.pairs:
            temp = df_norm[pair[0]] - df_norm[pair[1]]
            temp_sq = temp * temp
            NPDs.append(temp_sq.sum())

        df_npd = pd.DataFrame([NPDs], columns=self.pairs)
        df_npd.index = ["NPD"]
        return df_npd.T

    def tls(self, p0, p1):

        def f(B, x):
            """Linear function y = m*x + b"""
            return B[0] + B[1] * x

        pair_y = self.log_df[p0]
        pair_x = self.log_df[p1]
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


class mrp:

    def __init__(self, df, n=50, adf_threshold=-2, target_variance=0.03, nlags=10):
        self.cigs = cig_subspace(df, n, adf_threshold)
        self.df = self.cigs.df
        self.beta = self.cigs.summary["B1"]
        self.spread = self.extract_spread(self.df)
        self.covs = self.autocov(nlags)
        self.spread_weights = self.minimize_port(target_variance)
        self.stock_weights = self.positioning(self.spread_weights)
        self.stocks = self.stock_weights.index.values

    def extract_spread(self, df):
        cig_pairs = self.cigs.cig_pairs
        st = pd.DataFrame()
        for i in range(len(cig_pairs)):
            y = df[cig_pairs.index[i][0]]
            x = df[cig_pairs.index[i][1]]
            b1 = self.cigs.summary.iloc[i, 1]
            s = y - b1 * x
            s.name = cig_pairs.index[i]
            st = pd.concat([st, s], axis=1)
        return st

    def autocov(self, nlags):
        model = VAR(self.spread)
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

    def minimize_port(self, target_variance):
        n = self.covs.shape[1]
        initialguess = np.repeat(1 / n, n)
        bounds = ((-1.0, 1.0),) * n
        weights_constraint = {
            'type': 'eq',
            'fun': lambda weights: np.sum(weights)
        }
        vol_constraint = {
            'type': 'eq',
            'fun': lambda weights: self.variance(weights, self.covs[0]) - target_variance ** 2
        }
        out = minimize(self.portmanteau,
                       initialguess,
                       method="SLSQP",
                       options={'disp': False},
                       constraints=[weights_constraint, vol_constraint],
                       bounds=bounds
                       )

        return pd.DataFrame(out.x, index=self.spread.columns)

    def positioning(self, spread_weights):
        """
        Make sure the beta's stock pairs is in the same order with the weights
        """
        long = spread_weights.copy(deep=True)
        short = -self.beta * spread_weights.T.values[0]
        long.index = [pair[0] for pair in long.index]
        short.index = [pair[1] for pair in short.index]
        out = pd.concat([long, short])
        out = out.groupby(out.index).sum()

        return out
