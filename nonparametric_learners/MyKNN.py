import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MyKNN():
    def __init__(self, k, knn_type='classification'):
        self.k = k
        self.type = knn_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = [self._pred_one(new_x) for new_x in X]
        return y_pred

    def _pred_one(self, pred_x):
        distances = np.linalg.norm(self.X_train - pred_x, axis=1)
        neighbors = distances.argsort()[:self.k]
        neighbors_targets = self.y_train[neighbors]
        if self.type == 'classification':
            (_, idx, counts) = np.unique(neighbors_targets,
                                         return_index=True,
                                         return_counts=True)
            index = idx[np.argmax(counts)]
            y_pred = neighbors_targets[index]
        else:
            y_pred = np.mean(neighbors_targets)
        return y_pred


def sim_data(num_obs):
    np.random.seed(123)
    n = num_obs

    x1 = np.random.multivariate_normal([0, 0], [[1, .4],[.4, 1]], n)
    x2 = np.random.multivariate_normal([1, 4], [[1, .6],[.6, 1]], n)

    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def plot_data(X, y):
    plt.figure(figsize=(12,8))
    plt.scatter(X[:,0], X[:,1], c=y, alpha=0.5)
    plt.show()

def main():
    X, y = sim_data(1000)
    plot_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = MyKNN(k=5)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    plot_data(X_train, y_pred_train)
    plot_data(X_test, y_test)
    plot_data(X_test, y_pred_test)

if __name__ == '__main__':
    main()
