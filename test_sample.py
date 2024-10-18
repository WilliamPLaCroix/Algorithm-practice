import sklearn as sk
from sklearn.linear_model import LinearRegression as sklr
import numpy as np
import pandas as pd
import linear_model as lm


import random
random.seed(42)


def test_linear_regression_fit():
    test_X = np.array([random.randint(-100, 100) for i in range(10)]).reshape(-1, 1)
    test_y = np.array([random.randint(-10, 10) for i in range(10)])
    model = lm.LinearRegression()
    model.fit(test_X, test_y)
    verification_model = sk.linear_model.LinearRegression()
    verification_model.fit(test_X, test_y)
    assert format(model.intercept, '.15f') == format(verification_model.intercept_, '.15f')
    assert format(model.coef, '.16f') == format(verification_model.coef_[0], '.16f')

# def test_linear_regression_coef():
#     test_X = np.array([random.randint(-100, 100) for i in range(10)]).reshape(-1, 1)
#     test_y = np.array([random.randint(-10, 10) for i in range(10)])
#     model = lm.LinearRegression()
#     model.fit(test_X, test_y)
#     verification_model = sk.linear_model.LinearRegression()
#     verification_model.fit(test_X, test_y)
#     assert format(model.coef, '.16f') == format(verification_model.coef_[0], '.16f')

