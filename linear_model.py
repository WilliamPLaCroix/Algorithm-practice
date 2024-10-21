import numpy as np
import pandas as pd

class LinearRegression:
    """
    Numpy implementation of 2d Linear Regression, with model fit, predict, and R2 score.
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fit regression model to X, y, minimizing OLS to estimate the slope and intercept.
        Currently only supports 2d (single-feauture) X.
        """
        # coef  = sum((x - x_mean)(y - y_mean)) / sum(x - x_mean)^2
        # intercept = y_mean - slope * x_mean
        self.X = np.array(X).copy().reshape(-1)
        self.y = np.array(y).copy()
        mean_X = np.mean(self.X)
        mean_y = np.mean(self.y)
        difference_X = self.X-mean_X
        difference_y = self.y-mean_y
        coef = np.sum(difference_X * difference_y) / np.sum(difference_X ** 2)
        self.coef_ = np.array([coef])
        self.intercept_ = mean_y - self.coef_.item() * mean_X
        return
    
    def score(self, X, y, weights=[]):
        if len(weights) == 0:
            weights = np.ones((len(y),1))
        else:
            raise NotImplementedError("Weights are not implemented")
        #print(weights)
        X = X.reshape(-1)
        weights = weights.reshape(10,)
        # #print("X", X.shape, "y", y.shape, "w", weights.shape)
        # #print(X)
        # print("X", X.shape, "y", y.shape, "w", weights.shape)
        # #X = X 
        # # split RSS steps
        # preds = self.predict(X)
        # print("preds", preds.shape)
        # pred_diff = (y - preds)
        # print("pred_diff", pred_diff.shape)
        # pred_squared = (pred_diff ** 2)
        # print("pred_squared", pred_squared.shape)
        # RSS = np.sum(pred_squared)
        # print("RSS", RSS)
        # # split TSS steps
        # y_diff = (y - np.mean(y))
        # print("y_diff", y_diff.shape)
        # y_squared = (y_diff ** 2)
        # print("y_squared", y_squared.shape)
        # TSS = np.sum(y_squared)
        #print("TSS", TSS)
        RSS = np.sum(((y - (self.predict(X))) ** 2))
        TSS = np.sum((((y - (np.mean(y)))) ** 2))
        score = 1 - (RSS/TSS)
        return score

    def predict(self, X):
        y_pred = X * self.coef_ + self.intercept_
        #print(y_pred.shape, y_pred)
        return y_pred.reshape(-1)
    
    def __repr__(self):
        return f"LinearRegression(coef={self.coef_}, intercept={self.intercept_}, score={self.score(self.X, self.y)})"