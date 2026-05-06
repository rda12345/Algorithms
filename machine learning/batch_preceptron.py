"""
Batch perceptron algorithm

Proposed by Rosenblatt in 1958, the batch perceptron algorithm is an online learning algorithm for binary classification.
The algorithm maintains a weight vector w, which is updated at each iteration, based on the examples which are misclassified by w. 
The algorithm is guaranteed to converge after at most R^2*B^2 iterations, where R is the radius of the smallest ball containing the data points,
and B is the margin of the data, min{||w||: for all i=1, ..., m, y_i(w^T x_i) >= 1}.
Complexity: Each iteration has a complexity of O(d) (taking an inner product), where d is the dimension of the data.
            Therefore, after T iterations, the total running time is O(T*d) = O(R^2*B^2*d).
"""
#TODO: verify the complexity, test the algorithm, something is wierd with the convergence guarantee, maybe it is R^2/B^2 instead of R^2*B^2
import numpy as np
import matplotlib.pyplot as plt

def batch_perceptron(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Returns the weight vector w, associated with the batch perceptron algorithm.

    Args:
        data (np.ndarray (d, m)): training data set
        labels (np.ndarray (m, )): training labels

    Returns:
        np.ndarray (d, 1): the weight vector w
    """
    d, m = data.shape
    w = np.zeros(d)
    converged = False
    while not converged:
        converged = True
        for i in range(m):
            if labels[i] * (w.T @ data[:, i]) <= 0:
                w += labels[i] * data[:, i]
                converged = False
    return w



if __name__ == "__main__":
    # generate linearly separable 2D data
    np.random.seed(42)
    m = 200

    X_pos = np.random.randn(2, m // 2) + np.array([[2], [2]])
    X_neg = np.random.randn(2, m // 2) + np.array([[-2], [-2]])

    X = np.hstack([X_pos, X_neg])
    y = np.hstack([np.ones(m // 2), -np.ones(m // 2)])

    idx = np.random.permutation(m)
    X, y = X[:, idx], y[idx]

    split = int(0.8 * m)
    data_train, data_test = X[:, :split], X[:, split:]
    labels_train, labels_test = y[:split], y[split:]
    # run the algorithm
    w = batch_perceptron(data_train, labels_train)

    # predict on test set
    predictions = [np.sign(w.T @ x) for x in data_test.T]
    # compute error
    error = np.mean(np.not_equal(predictions, labels_test))
    print(f"Test error: {error}")

    # plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_pos[0], X_pos[1], c='red', marker='o', label='Positive')
    plt.scatter(X_neg[0], X_neg[1], c='blue', marker='x', label='Negative')
    plt.legend()
    plt.title('Training Data')

    plt.subplot(1, 2, 2)
    plt.scatter(X_pos[0], X_pos[1], c='red', marker='o', label='Positive')
    plt.scatter(X_neg[0], X_neg[1], c='blue', marker='x', label='Negative')
    # plot the decision boundary
    x_line = np.linspace(-4, 4, 100)
    y_line = -(w[0] * x_line) / w[1]
    plt.plot(x_line, y_line, 'k--', label='Decision Boundary')
    plt.legend()
    plt.title('Test Data')

    plt.show()