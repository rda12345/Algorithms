"""
Halving learning algorithm
The halving algorithm is an online learning algorithm which maintains a set of "consistent" hypotheses,
which are those that are correct on all the examples seen so far.
At each step, the algorithm predicts according to the majority vote of the consistent hypotheses.
If the prediction is correct, the set of consistent hypotheses remains unchanged.
Otherwise, all the inconsistent hypotheses are removed from the set of consistent hypotheses.

The algorithm is guaranteed to converge after at most log_2(|H|) mistakes, where H is the set of hypotheses.
Complexity: O(|H|*N), where N is the number of examples.

We test the halving learner on a simple threshold problem, where where if the data points are in the interval [0,1],
and the labels are 1 if the 300th element of the data point is greater than 0.5, and -1 otherwise.
The hypotheses class consists of discrete threshold functions on the real line.

Halving learner variant for discrete threshold functions on the real line.
The halving learner can be applied efficiently to a simple hypothesis class, 
which consists of all the discrete threshold functions on the real line.

Complexity: O(log(N)), where N is the number of examples, and d is the dimension of the data.
"""
import numpy as np
from typing import Callable

def halving(data: np.ndarray, labels: np.array, hypotheses: list) -> Callable | None:
    """
    Returns a consistent hypothesis, if it exists, otherwise returns None.

    Args:
        data (np.ndarray (N, d)), the columns of data constitute the data points
        labels (np.array (N,)), the labels of the data points
        hypotheses (list of callables), the set of hypotheses to choose from
    
    Returns:
        callable: a consistent hypothesis, if it exists, otherwise None
    """
    N = data.shape[0]
    for t in range(N):
        # predict according to the majority vote of the consistent hypotheses
        predictions = [h(data[t]) for h in hypotheses]  
        majority_vote = np.sign(np.sum(predictions))
        if majority_vote != labels[t]:
            # remove inconsistent hypotheses
            hypotheses = [h for h in hypotheses if h(data[t]) == labels[t]]
    if hypotheses:
        return hypotheses[0]  # return an arbitrary consistent hypothesis
    else:
        return None
    

def halving_descrite_threshold(data: np.ndarray, labels: np.array, n: int) -> Callable | None:
    """
    The halving learner can be applied efficiently to a hypothesis class,
    which consists of all the discrete threshold functions on the real line.
    The data points are assumed to be in the interval [0,1],
    and the hypotheses are of the form h_i(x) = 1 if x >= i/n, and 0 otherwise.

    Args:
        data (np.ndarray (N,)), the data points
        labels (np.array (N,)), the labels of the data points
        n (int): the number of discrete thresholds
    
    Returns:
        l (float): left endpoint of the interval
        r (float): right endpoint of the interval
    """
    N = data.shape[0]
    l = -0.5/n
    r = 1 + 0.5/n
    for t in range(N):
        x = data[t]
        y = labels[t]
        if (x >= l and x <= r) and (r - l) > 1/n:
            if y == 1:
                r = x + 0.5/n
            else:
                l = x + 0.5/n
    return l, r

def predict(l, r, x):
    """
    Predicts the label of x, based on the interval [l, r].

    Args:
        l (float): left endpoint of the interval
        r (float): right endpoint of the interval
        x (float): the point to predict

    Returns:
        int: the predicted label (1 or -1)
    """
    return np.sign((x - l) - (r - x))

if __name__ == "__main__":
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
    h = halving(data_train, labels_train, hypotheses)
    # predict on test set
    if h is not None:
        predictions = [h(data_test[i,:]) for i in range(data_test.shape[0])]
        error = np.mean(np.not_equal(predictions, labels_test))
        print(f"Test error of threshold problem (batch learning): {error}")
    else:        print("No consistent hypothesis found (batch learning).")


    ## halving learner for discrete threshold functions on the real line
    data = np.linspace(0, 1,100, endpoint=False)
    labels = np.where(data >= 0.6, 1, -1)
    # split data into train and test sets
    N = len(labels)
    split = int(0.8 * N)    # 80% for training, 20% for testing
    data_train = data[:split]
    labels_train = labels[:split]
    data_test = data[split:]
    labels_test = labels[split:]
    l, r = halving_descrite_threshold(data_train, labels_train, n=100)
    predictions = [predict(l, r, x) for x in data_test]
    error = np.mean(np.not_equal(predictions, labels_test))
    print(f"Test error of threshold problem (halving learner for discrete threshold functions): {error}")