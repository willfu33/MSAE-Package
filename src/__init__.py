import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def plot_linreg6(dataset, lr_model, feat_num):
# input: dataset, a trained linear regression model lr_model, and feat_num: which feature number of X you want to plot against y target
# output: a plot of the x and y data points as well as a fitted linear regression line.
    slope = lr_model.coef_
    intercept = lr_model.intercept_

    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    xvec = X.T[feat_num, :] #takes all the values of the feat_num column of X
    
    xline = np.linspace(min(X_train), max(X_train), 100)
    y_fit = intercept + slope * xline
    plt.plot(xvec, y, "o")
    plt.plot(xline, y_fit, "--")  # this is the linear regression line

def train_linreg6(dataset):
# input: dataset (example: mglearn.datasets.make_wave(n_samples=60) )
# output: returns lr, the trained model
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LinearRegression().fit(X_train, y_train)
    return lr