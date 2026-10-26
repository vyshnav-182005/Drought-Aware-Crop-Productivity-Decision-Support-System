"""
Shared baseline model class for yield modeling.
Predicts from district historical mean yield + simple linear trend.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

BASELINE_FEATURES = ["district_mean_yield", "trend"]


class BaselineModel(RegressorMixin, BaseEstimator):
    def fit(self, X_df, y):
        self.reg_ = LinearRegression().fit(X_df[BASELINE_FEATURES], y)
        return self

    def predict(self, X_df):
        return self.reg_.predict(X_df[BASELINE_FEATURES])
