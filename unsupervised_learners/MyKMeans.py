import numpy as np
import matplotlib.pyplot as plt

class MyKMeans():
    def __init__(self, k, max_iter=1000):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X, y=None):
        n = X.shape[0]
        initial_ind = np.random.randint(0, n, self.k)
        self.centroids = X[initial_ind]
        labels = np.zeros(n)
        for _ in range(self.max_iter):
            for i, x in enumerate(X):
                distances = np.linalg.norm(self.centroids - x, axis=1)
                labels[i] = np.argmin(distances)

            for lab in range(self.k):
                cluster = [row for i, row in enumerate(X) if labels[i] == lab]
                self.centroids[lab] = np.mean(cluster, axis=0)

        return self

    def predict(self, X):
        n = X.shape[0]
        labels = np.zeros(n)
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.centroids - x, axis=1)
            labels[i] = np.argmin(distances)
        return labels

def sim_data(num_obs):
    np.random.seed(123)
    n = num_obs

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.8],[0.8, 1]], n)
    x2 = np.random.multivariate_normal([1, 4.5], [[1, .8],[.8, 1]], n)
    x3 = np.random.multivariate_normal([4.5, 1.5], [[1, .8],[.8, 1]], n)

    X = np.vstack((x1, x2, x3)).astype(np.float32)
    return X

def plot_data_wcentroids(X, centroids_old, centroids_new):
    plt.figure(figsize=(12,8))
    plt.scatter(X[:,0], X[:,1], alpha=0.5)
    plt.scatter(centroids_old[:,0], centroids_old[:,1], marker='o', s=200, c='#050505', alpha=0.5)
    plt.scatter(centroids_new[:,0], centroids_new[:,1], marker='*', s=200, c='#050505', alpha=0.5)
    plt.show()

def plot_data(X):
    plt.figure(figsize=(12,8))
    plt.scatter(X[:,0], X[:,1], alpha=0.5)
    plt.show()

def main():
    X = sim_data(100)
    np.random.shuffle(X)
    X_train = X[:275]
    X_test = X[275:]
    plot_data(X_train)
    plot_data(X_test)

    kmeans = MyKMeans(k=3)
    kmeans.fit(X_train)
    lab_pred_train = kmeans.predict(X_train)
    lab_pred_test = kmeans.predict(X_test)

    plt.figure(figsize=(12,8))
    plt.scatter(X_train[:,0], X_train[:,1], c=lab_pred_train, alpha=0.5)
    plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], marker='*', c='black', alpha=0.5)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.scatter(X_test[:,0], X_test[:,1], c=lab_pred_test, alpha=0.5)
    plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], marker='*', c='black', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()
