# -*- coding: utf-8 -*-
"""
Created on Thur Apr 6 2018

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

X = np.array(pd.read_csv("X.csv", header=None), dtype=np.float64)
y = np.array(pd.read_csv("y.csv", header=None), dtype=np.float64)


class Elastic_Net:

    def __init__(self, alpha, l1_ratio=1.0, max_iter=1000, eps=1e-4, normalize=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.eps = eps
        self.coef_ = None
        self.normalize = normalize
        self.X_scaled = None

    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0

    def fit_transform(self, X):
        if self.normalize:
            self.X_scaled = scale(X)

    def fit(self, X, y):
        self.fit_transform(X)
        beta = np.zeros((self.X_scaled.shape[1], 1))
        beta_list = [1] * (self.max_iter)
        beta_list[0] = beta
        for i in range(self.max_iter):
            for j in range(len(beta)):
                tmp_beta = deepcopy(beta)
                tmp_beta[j] = 0.0
                r_j = y - np.dot(self.X_scaled, tmp_beta)
                arg1 = np.dot(r_j.T, self.X_scaled[:, j])[0]
                arg2 = self.alpha * self.l1_ratio * self.X_scaled.shape[0]
                beta[j] = self._soft_thresholding_operator(arg1, arg2) / ((self.X_scaled[:, j] ** 2).sum()) / (1 + self.alpha * (1 - self.l1_ratio))
            
            beta_list[i+1] = beta
            if np.linalg.norm(beta_list[i] - beta, ord=2) < self.eps and i > 0:
                break
        self.coef_ = beta
        return self

    def predict(self, X):
        y = np.dot(X, self.coef_)
        return y

if __name__ == "__main__":
    alphas = [0.01, 0.1, 1, 10]
    for _, v in enumerate(alphas):
        model = Elastic_Net(alpha=v, l1_ratio=0.95)
        model.fit(X, y)
        print(mean_squared_error(model.predict(X), y))