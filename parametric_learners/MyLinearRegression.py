# This function implements standard linear regression using gradient descent
# of the errors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class MyLinearRegression():
    def __init__(self, num_steps, learning_rate, add_intercept):
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.add_intercept = add_intercept

    def fit(self, X, y):
        n = X.shape[0]
        if self.add_intercept:
            intercept = np.ones((n, 1))
            X = np.hstack((intercept, X))

        self.weights_ = np.zeros(X.shape[1])

        for step in range(self.num_steps):
            predictions = np.dot(X, self.weights_)
            error = y - predictions
            gradient = np.dot(X.T, error)
            self.weights_ += self.learning_rate * gradient

            # predictions = np.dot(X, self.weights_)
            # cost = np.sum((y - predictions)**2)/n
            # gradient = -(2/n) * np.sum(y - predictions)
            # self.weights_ -= self.learning_rate * gradient

        return self

    def predict(self, X):
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        return np.dot(X, self.weights_)

    def score(self, X, y):
        '''
        default R2
        '''
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        y_pred = np.dot(X, self.weights_)
        e = y - y_pred
        SSres = np.sum(e**2)
        SStot = np.sum((y - np.mean(y))**2)
        return 1 - (SSres/SStot)

def sim_data(num_obs):
    np.random.seed(123)
    n = num_obs
    X = np.linspace(-3,3,n)
    y = np.sin(X) + np.random.uniform(-0.5,0.5,n)
    return X.reshape((n,1)), y

def plot_data(X, y):
    plt.figure(figsize=(12,8))
    plt.scatter(X, y, alpha=0.5)
    plt.axis([-4,4,-2.0,2.0])
    plt.show()

def main():
    X, y = sim_data(1000)
    plot_data(X, y)

    mylinreg = MyLinearRegression(num_steps = 200000,
                                    learning_rate=5e-5,
                                    add_intercept=True)
    mylinreg.fit(X, y)

    sklinreg = LinearRegression()
    sklinreg.fit(X, y)

    print('Weights - MyLinReg, {}'.format(mylinreg.weights_))
    print('Weights - SKLearn, {}, {}'.format(sklinreg.intercept_, sklinreg.coef_))

    print('Accuracy - MyLinReg, {}'.format(mylinreg.score(X, y)))
    print('Accuracy - SKLearn, {}'.format(sklinreg.score(X, y)))

if __name__ == '__main__':
    main()
