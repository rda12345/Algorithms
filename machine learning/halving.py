"""
Halving learning algorithm
The halving algorithm is an online learning algorithm which maintains a set of "consistent" hypotheses,
which are those that are correct on all the examples seen so far.
At each step, the algorithm predicts according to the majority vote of the consistent hypotheses.
If the prediction is correct, the set of consistent hypotheses remains unchanged.
Otherwise, all the inconsistent hypotheses are removed from the set of consistent hypotheses.

The algorithm is guaranteed to converge after at most log_2(|H|) mistakes, where H is the set of hypotheses.
Complexity: O(|H|*N), where N is the number of examples.

Halving learner variant for discrete threshold functions on the real line.
The halving learner can be applied efficiently to a simple hypothesis class, 
which consists of all the discrete threshold functions on the real line.

Complexity: O(log(N)), where N is the number of examples, and d is the dimension of the data.
"""
import numpy as np

def halving(data: np.ndarray, labels: np.array, hypotheses: list) -> callable | None:
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
        predictions = [h(x) for h in hypotheses for x in data[t]]  
        majority_vote = np.sign(np.sum(predictions))
        if majority_vote != labels[t]:
            # remove inconsistent hypotheses
            hypotheses = [h for h in hypotheses if h(data[:,t]) == labels[t]]
    if hypotheses:
        return hypotheses[0]  # return an arbitrary consistent hypothesis
    else:
        return None
    

def halving_descrite_threshold(data: np.ndarray, labels: np.array, n: int) -> callable | None:
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
    l = -0.5/n
    r = 1 + 0.5/n
    for t in range(N):
        x = data[t]
        y = labels[t]
        if x >= l and x <= r:
            if y == 1:
                r = x - 0.5/n
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
    return np.sign((x - l) * (x - r))

if __name__ == "__main__":
    # hypotheses classes
    hypotheses = 

    # uplaod data set
    data_train =        # training set
    data_test =         # test set
    labels_train =   # training labels
    labels_test =    # test labels
    # run the algorithm
    h = halving(data_train, labels_train, hypotheses)

    # predict on test set
    predictions = [h(x) for x in data_test.T]
    # compute error
    error = np.mean(np.not_equal(predictions, labels_test))
    print(f"Test error: {error}")


    # TDOD: run the halving learner for discrete threshold functions on the real line