"""
This script allows to perform a simple linear regression fit on the input
X: predictor of shape (N,)
Y: response  of shape (N,)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   scipy import stats

np.set_printoptions(legacy='1.25')

class simple_linear_regression:
    def __init__(self, X: np.array, Y: np.array):
        self.X = X
        self.Y = Y

        self._check_dims(X = self.X, Y = self.Y)
        self.fit()

    @staticmethod
    def _check_dims(X: np.array, Y: np.array) -> None:
        """Check dimensions of X and Y match"""
        if X.values.flatten().shape[0] == Y.values.flatten().shape[0]: pass
        else: raise ValueError(f"X and Y of different shape: {X.shape} vs {Y.shape}")

    def fit(self) -> None:
        """Perform simple linear regression fit using least squares approach"""
        df = pd.DataFrame({'X': self.X, 'Y': self.Y})
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        self.n_data = int(df['X'].count())
        self.dof    = self.n_data - 2  # two coefficients are estimated from the data (inter, slope)

        # Evaluate the coefficient estimators from the fit
        x_mean = df['X'].mean()
        y_mean = df['Y'].mean()

        df['X_center']    = df['X'] - x_mean
        df['Y_center']    = df['Y'] - y_mean
        df['XY']          = df['X_center'] * df['Y_center']
        Sxy               = df['XY'].sum()
        df['X_center_sq'] = df['X_center'] ** 2
        Sxx               = df['X_center_sq'].sum()
        df['Y_center_sq'] = df['Y_center'] ** 2
        Syy               = df['Y_center_sq'].sum()

        # Coefficient estimators (unbiased)
        slope = Sxy / Sxx
        inter = y_mean - slope * x_mean
        self.coeffs = [inter, slope]
        
        df['Y_pred'] = inter + slope * df['X']
        
        # Sum of squares
        df['tss'] = (df['Y']      - y_mean) ** 2
        df['ess'] = (df['Y_pred'] - y_mean) ** 2
        df['rss'] = (df['Y']      - df['Y_pred']) ** 2
        TSS = df['tss'].sum() # total in the data
        ESS = df['ess'].sum() # explained by model
        RSS = df['rss'].sum() # unexplained (residual)
        # print(f'Verify that TSS = ESS + RSS = {ESS} + {RSS} = {ESS + RSS} [{TSS}]')

        # R2 statistic
        self.r2  = (TSS - RSS) / TSS
        
        # Standard error estimators
        s2       = RSS / self.dof # estimator for variance of epsilon (irreducible error term)
        slope_se = np.sqrt( s2 / Sxx )
        inter_se = np.sqrt( s2 * (1/self.n_data + x_mean**2 / Sxx) )
        cov_     = (-1) * s2 * x_mean / Sxx
        self.se  = np.array([[inter_se, cov_],
                             [cov_, slope_se]])

        # Correlation 
        self.corr = Sxy / np.sqrt( Sxx * Syy)

        # t-test
        t_slope = (slope - 0) / slope_se
        t_inter = (inter - 0) / inter_se
        self.t = [t_inter, t_slope]

        self.p_value = []
        for t in self.t:
            p = 2 * stats.t.sf(abs(t), self.dof)
            p = max(0.0, min(1.0, p))
            self.p_value.append(p)

        results = {'coeffs': self.coeffs,
                   'errors': self.se,
                   'TSS': TSS,
                   'ESS': ESS,
                   'RSS': RSS,
                   'r2': self.r2,
                   'corr': self.corr, 
                   't': self.t,
                   'dof': self.dof,
                   'p': self.p_value,
        }
        self.results = results

    

