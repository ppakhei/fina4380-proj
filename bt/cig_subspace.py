import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
import itertools

class cig_subspace():

    def __init__(self, df, n=50, adf_threshold = -2):
        self.df = df
        self.n = n
        self.adf_threshold = adf_threshold
        self.pairs = list(itertools.combinations(df.columns, 2))
        df_npd = self.normalized_price_distance(self.pairs, self.df)
        npd_pairs = df_npd.nsmallest(self.n, "NPD")
        npd_pairs = npd_pairs.index.tolist()

        beta, df_adf = self.adf(npd_pairs, self.df)
        self.beta = beta
        self.cig_pairs = df_adf[df_adf < self.adf_threshold].dropna().sort_values(by="ADF")
        self.summary = pd.concat([self.beta, self.cig_pairs], axis=1).dropna(axis=0).sort_values(by="ADF")

    def normalized_price_distance(self, pairs, df):
        df_norm = df/df.iloc[0, :]
        NPDs = []
        for pair in pairs:
            temp = df_norm[pair[0]] - df_norm[pair[1]]
            temp_sq = temp*temp
            NPDs.append(temp_sq.sum())

        df_npd = pd.DataFrame([NPDs], columns=pairs)
        df_npd.index = ["NPD"]
        return df_npd.T

    def tls(self, y, x):

        def f(B, x):
            '''Linear function y = m*x + b'''
            return B[0] + B[1] * x

        linreg = linregress(x, y)
        mod = Model(f)
        dat = Data(x, y)
        od = ODR(dat, mod, beta0=linreg[0:2])
        out = od.run()
        return out.beta

    def adf(self, npd_pairs, df):

        adf_stats = []
        b0s = []
        b1s = []
        for pair in npd_pairs:
            y = df[pair[0]]
            x = df[pair[1]]
            b0, b1 = self.tls(y, x)
            resid = (y - b0 - b1*x)/(np.sqrt(1+b1*b1))
            adf_stat = adfuller(resid)[0]
            b0s.append(b0)
            b1s.append(b1)
            adf_stats.append(adf_stat)

        df_beta = pd.DataFrame([b0s, b1s], columns=npd_pairs)
        df_beta.index = ["B0", "B1"]
        df_adf = pd.DataFrame([adf_stats], columns=npd_pairs)
        df_adf.index = ["ADF"]
        return df_beta.T, df_adf.T
