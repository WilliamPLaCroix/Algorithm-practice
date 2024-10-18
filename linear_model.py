import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.score = None
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        # coef  = sum((x - x_mean)(y - y_mean)) / sum(x - x_mean)^2
        # intercept = y_mean - slope * x_mean
        X = np.array(X).reshape(-1)
        mean_X = np.mean(X)
        mean_y = np.mean(y)
        # difference_X = X-mean_X
        # difference_y = y-mean_y
        # self.coef = np.sum(difference_X * difference_y) / np.sum(difference_X ** 2)
        # self.intercept = mean_y - self.coef * mean_X
        difference_X = X-mean_X
        difference_y = y-mean_y
        multiply = difference_X * difference_y
        sum_multiply = np.sum(multiply)
        bottom = np.sum(difference_X ** 2)
        self.coef = sum_multiply / bottom
        self.intercept = mean_y - self.coef * mean_X
        return

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError
    
    def __repr__(self):
        return f"LinearRegression(coef={self.coef}, intercept={self.intercept})"