from cig_subspace import cig_subspace
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.optimize import minimize

class mrp():

    def __init__(self, df, target_variance = 0.03, nlags = 10, NAV = 1, n=50, adf_threshold = -2):

        self.target_variance = target_variance
        self.NAV = NAV

        result = cig_subspace(df=df, n=n, adf_threshold = adf_threshold)
        beta = result.summary["B1"]
        log_df_close = np.log(df)

        self.st = self.extract_spread(result, log_df_close)
        self.M = self.autocov(self.st, nlags)
        self.weights = self.minimize_port(self.M, self.target_variance)
        self.position = self.positioning(beta, self.weights, self.NAV)


    def extract_spread(self, cointegrated_subspace, log_prices):
        """

        :param cointegrated_subspace:
        :param log_prices: log_prices
        :return:
        """
        cig_pairs = cointegrated_subspace.cig_pairs
        st = pd.DataFrame()
        for i in range(len(cig_pairs)):
            y = log_prices[cig_pairs.index[i][0]]
            x = log_prices[cig_pairs.index[i][1]]
            b1 = cointegrated_subspace.summary.iloc[i, 1]
            s = (y - b1*x)/np.sqrt(1+b1*b1)
            s.name = cig_pairs.index[i]
            st = pd.concat([st, s], axis=1)
        return st

    def autocov(self, spread, nlags):
        model = VAR(spread)
        M = model.fit(maxlags=nlags).sample_acov(nlags=nlags)
        return M

    def variance(self, weights, M):
        return weights.T@M@weights

    def portmanteau(self, weights, M):
        M0 = M[0]
        out = 0
        for m in M[1:]:
            v = self.variance(weights, m) / self.variance(weights, M0)
            out = out + v*v
        return out

    def minimize_port(self, M, target_variance = 0.03):
        n = M.shape[1]
        initialguess = np.repeat(1/n, n)
        bounds = ((-1.0,1.0),)*n
        weights_equal_1 = {
            'type': 'eq',
            'fun': lambda weights: np.sum(weights)
        }
        vol_level = {
            'type': 'eq',
            'fun': lambda weights: self.variance(weights, M[0]) - target_variance**2
        }
        out = minimize(self.portmanteau,
                           initialguess,
                           args=(M,),
                           method = "SLSQP",
                           options = {'disp': False},
                           constraints = (weights_equal_1, vol_level),
                           bounds = bounds
                          )
        weights = out.x
        weights = weights / (2 * weights[weights > 0].sum())
        return weights

    def positioning(beta, weights, NAV=1):
        """
        Make Sure the beta's stock pairs is in the same order with the weights
        :param beta:
        :param weights:
        :param NAV:
        :return:
        """
        long = pd.DataFrame([weights], columns=beta.index).T
        short = -beta * weights
        long.index = [seq[0] for seq in long.index]
        short.index = [seq[1] for seq in short.index]
        out = pd.concat([long, short]).sum(level=0)
        out = out / out.sum()

        return out / abs(out).sum() * NAV