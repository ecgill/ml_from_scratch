# This function implements standard logistic regression from scratch
# Logistic regression is a GLM with link=sigmoid and is used to predict a
# categorical target. To implement: find the weights that maximize the
# likelihood of producing our given data and use them to categorize the
# response variable --> MLE.
# Optimization: Gradient ascent to maximize the likelihood function.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class MyLogisticRegression():
    def __init__(self, num_steps, learning_rate, add_intercept):
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.add_intercept = add_intercept

    def _sigmoid(self, scores):
        '''
        Converts scores to values between 0 and 1 using sigmoid link fxn.
        '''
        return 1/(1 + np.exp(-scores))

    def _log_likelihood(self, X, y, weights):
        '''
        logL = sum(yi*Beta^T*xi - log(1 + e^(B^T*xi)))
        '''
        scores = np.dot(X, weights)
        ll = np.sum(y*scores - np.log(1+np.exp(scores)))
        return ll

    def fit(self, X, y):
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))

        self.weights_ = np.zeros(X.shape[1])

        for step in range(self.num_steps):
            scores = np.dot(X, self.weights_)
            predictions = self._sigmoid(scores)

            # Update weights with Gradient
            error = y - predictions
            gradient = np.dot(X.T, error)
            self.weights_ += self.learning_rate * gradient

            # Print log-likelihood once every so often:
            # if step % 10000 == 0:
            #     print(self._log_likelihood(X, y, self.weights_))
            #     print('\n', self.weights_, gradient)

        return self

    def predict(self, X):
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        scores = np.dot(X, self.weights_)
        return np.round(self._sigmoid(scores))

    def score(self, X, y):
        '''
        default Accuracy
        '''
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        scores = np.dot(X, self.weights_)
        y_pred = np.round(self._sigmoid(scores))
        return (y_pred == y).sum().astype(float) / len(y_pred)

def sim_data(num_obs):
    np.random.seed(123)
    n = num_obs

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], n)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], n)

    sim_feat = np.vstack((x1, x2)).astype(np.float32)
    sim_labels = np.hstack((np.zeros(n), np.ones(n)))
    return sim_feat, sim_labels

def plot_data(X, y):
    plt.figure(figsize=(12,8))
    plt.scatter(X[:,0], X[:,1], c=y, alpha=0.5)
    plt.show()

def main():
    X, y = sim_data(5000)
    plot_data(X, y)

    mylogreg = MyLogisticRegression(num_steps = 200000,
                                    learning_rate=5e-5,
                                    add_intercept=True)
    mylogreg.fit(X, y)

    sklogreg = LogisticRegression(fit_intercept=True, C=1e15)
    sklogreg.fit(X, y)

    print('Weights - MyLogReg, {}'.format(mylogreg.weights_))
    print('Weights - SKLearn, {}, {}'.format(sklogreg.intercept_, sklogreg.coef_))

    print('Accuracy - MyLogReg, {}'.format(mylogreg.score(X, y)))
    print('Accuracy - SKLearn, {}'.format(sklogreg.score(X, y)))



if __name__ == '__main__':
    main()
