import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from util import get_data as get_mnist


def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1

    rng = np.random.default_rng()
    X = rng.random((300, 2)) * 2 - 1
    Y = np.sign(X @ w + b)

    return X, Y


def get_simple_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    return X, Y


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        D = X.shape[1]

        rng = np.random.default_rng()
        self.w = rng.standard_normal(D)
        self.b = 0.0

        N = len(Y)
        costs = []

        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.flatnonzero(Y != Yhat)

            if incorrect.size == 0:
                break

            i = rng.choice(incorrect)

            self.w += learning_rate * Y[i] * X[i]
            self.b += learning_rate * Y[i]

            cost = incorrect.size / N
            costs.append(cost)

        print(f"final w: {self.w}, final b: {self.b}, epochs: {epoch + 1}/{epochs}")

        plt.plot(costs)
        plt.title("Training error")
        plt.show()

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == "__main__":
    # linearly separable data
    X, Y = get_data()
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()

    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Perceptron()

    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", datetime.now() - t0)

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", datetime.now() - t0)

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", datetime.now() - t0)

    # MNIST (0 vs 1)
    X, Y = get_mnist()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1

    model = Perceptron()

    t0 = datetime.now()
    model.fit(X, Y, learning_rate=1e-2)
    print("MNIST train accuracy:", model.score(X, Y))

    # XOR data
    print("\nXOR results:")
    X, Y = get_simple_xor()
    Y[Y == 0] = -1

    model.fit(X, Y)
    print("XOR accuracy:", model.score(X, Y))