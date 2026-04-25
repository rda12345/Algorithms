"""
The consistent learner algorithm

This is a basic learning algorithm, which assumes realizability, i.e.,
there is a hypothesis in the set of hypotheses which is correct on all the examples.
Assuming that the hypotheses class is finite, the algorithm maintains a set of "consistent" hypotheses,
which are those that are correct on all the examples seen so far.
Two versions of the algorithm are presented:
1. An online version, which predicts according to an arbitrary consistent hypothesis at each step.
2. A batch version, which returns an arbitrary consistent hypothesis after seeing all the examples.

At each step, the algorithm predicts according to an arbitrary consistent hypothesis.
If the prediction is correct, the set of consistent hypotheses remains unchanged.
Otherwise, all the inconsistent hypotheses are removed from the set of consistent hypotheses.

The algorithm is guaranteed to converge after at most |H|-1 mistakes, where H is the set of hypotheses.
Complexity: O(|H|*N), where N is the number of examples.

We test the online and batch versions of the algorithm on two corresponding problems:
1. A simple threshold problem, where the data points are in the interval [0,1], and the labels are 1 if the data point is greater than 0.5, and -1 otherwise.
 The hypotheses class consists of all the discrete threshold functions on the real line.
2. A digit classification problem, where the data points are images of digits, and the labels are 1 if the digit is 7, and 0 if the digit is 4.
 The hypotheses class consists of threshold functions on each pixel, i.e., h_i(x) = 1 if x[i] >= t, and 0 otherwise, for some threshold t.
"""
import numpy as np
from typing import Callable
import pandas
def consistent_online(data: np.ndarray, labels: np.array, hypotheses: list, realizable: bool = True) -> Callable | None:
    """
    Returns a consistent hypothesis, if it exists, otherwise returns None.

    Args:
        data (np.ndarray (N, d)), the columns of data constitute the data points
        labels (np.array (N,)), the labels of the data points
        hypotheses (list of callables), the set of hypotheses to choose from
        realizable (bool), whether the realizability assumption holds

    Returns:
        callable: a consistent hypothesis, if it exists, otherwise None
    """
    N, _ = data.shape
    for t in range(N):
        x = data[t]
        y = labels[t]
        # predict according to an arbitrary consistent hypothesis
        if hypotheses:
            h = hypotheses[0]  # arbitrary consistent hypothesis
            y_pred = h(x)
            if y_pred != y:
                # remove inconsistent hypotheses
                hypotheses = [h for h in hypotheses if h(x) == y]
        else:
            continue
    if hypotheses:
        return hypotheses[0]  # return an arbitrary consistent hypothesis
    elif not realizable:  # return the last hypothesis, which is the best one, even if it is not consistent
        return h
    else: 
        return None

    
def consistent_batch(data: np.ndarray, labels: np.array, hypotheses: list) -> Callable | None:
    """
    Returns a consistent hypothesis, if it exists, otherwise returns None.

    Args:
        data (np.ndarray (N, d)), the columns of data constitute the data points
        labels (np.array (N,)), the labels of the data points
        hypotheses (list of callables), the set of hypotheses to choose from

    Returns:
        callable: a consistent hypothesis, if it exists, otherwise None
    """
    consistent_hypotheses = []
    for h in hypotheses:
        if all(h(x) == y for x, y in zip(data, labels)):
            consistent_hypotheses.append(h)
    if consistent_hypotheses:
        return consistent_hypotheses[0]  # return an arbitrary consistent hypothesis
    else:
        return None
    

if __name__ == "__main__":    
    from sklearn.datasets import fetch_openml

    # Points are +1 if pixel 300 > 0.5, else -1
    data = np.random.rand(1000, 784)
    labels = np.where(data[:, 300] > 0.5, 1, -1)

    # split data into train and test sets
    N = len(labels)
    split = int(0.8 * N)    # 80% for training, 20% for testing
    data_train = data[:split]
    labels_train = labels[:split]
    data_test = data[split:]
    labels_test = labels[split:]
    
    # hypotheses class consists of threshold functions on each pixel, i.e., h_i(x) = 1 if x[i] >= 0.5, and 0 otherwise
    thresholds = [0.5]
    hypotheses = [lambda x, i=i, t=t: 1 if x[i] > t else -1 for i in range(data.shape[1]) for t in thresholds]

    # run the batch version of the algorithm
    h = consistent_batch(data_train, labels_train, hypotheses)
    # predict on test set
    if h is not None:
        predictions = [h(data_test[i,:]) for i in range(data_test.shape[0])]
        error = np.mean(np.not_equal(predictions, labels_test))
        print(f"Test error of threshold problem (batch learning): {error}")
    else:        print("No consistent hypothesis found (batch learning).")



    #Load MNIST data set and filter digits 4 and 7
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target']
    mask  = (y == '4') | (y == '7')
    data = X[mask]
    labels = y[mask]
    labels = np.where(labels == '4', 0, 1)  # convert
    data = data / 255.0 # normalize data to [0,1]
    thresholds = [0.3, 0.5, 0.7]
    hypotheses = [lambda x, i=i, t=t: 1 if x[i] > t else 0 for i in range(data.shape[1]) for t in thresholds]


    # split data into train and test sets
    N = len(labels)
    split = int(0.8 * N)    # 80% for training, 20% for testing
    data_train = data[:split]
    labels_train = labels[:split]
    data_test = data[split:]
    labels_test = labels[split:]

    
    # run the online version of the algorithm, without realizability assumption
    h = consistent_online(data_train, labels_train, hypotheses, realizable=False)
     # predict on test set
    predictions = [h(data_test[i,:]) for i in range(data_test.shape[0])]
    if predictions:
        error = np.mean(np.not_equal(predictions, labels_test))
        print(f"Test error of the digit classification problem (online learning): {error:.5f}")
    else:        print("No consistent hypothesis found for the digit classification problem (online learning).")





    

