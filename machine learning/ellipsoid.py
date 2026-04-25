"""
Ellipsoid learner algorithm

An online learning algorithm which maintains an ellipsoid that contains the V_t, the set of hypotheses
which are correct on all the t first examples x_1,...,x_{t-1}.
Assumes realizability, i.e., there is a linear separator in the set of hypotheses,
and that the data is linearly separable.
Each mistake reduces the volume of the ellipsoid, E_t, by a factor e^{-1/(2*d+2)}.
Moreover, the volume of E_t is bounded by Vol(B) * (1/n)^{2*d}.
????????
, therefore it converges rapidly to the best hypothesis.

The algorithm is guaranteed to converge after at most (2*d+2)*log(R) mistakes,
where R is the radius of the smallest ball containing the set of hypotheses.
Complexity:
    
"""
import numpy as np
#TODO complete example
def ellipsoid(data: np.ndarray, labels: np.ndarray) -> callable:
    """
    The ellipsoid algorithm maintains implicitly maintains an ellipsoid
    eps_t = E(sqrt(A_t), w_t), where E(B, u) is an ellipsoid, created by
    the affine transformation B*x + u, where x are the points forming a unit ball.
    A_t and w_t, are the associated matrices at step t.

    Args:
        data (np.ndarray (d,N)): training data set
        labels (np.ndarray (N,))
    
    Returns:
        w (np.ndarray (d,)): bais of the converged ellipsoid
    """
    d, N = data.shape
    w = np.zeros(d)
    A = np.eye(d)
    for t in range(N):    
        x = data[:,t]
        y = labels[t]
        y_pred = np.sign(np.dot(w, x))   # prediction of y
        if y_pred != y:
            temp = A@x/(np.sqrt(x.T @ A @ x))
            w += (y/(d+1)) * temp 
            A =  (d^2/(d^2-1))*(A - (2/(d+1))*(np.outer(temp, temp)))
        
def predict(w, x):
    """
    Predicts the label of x, based on the ellipsoid defined by A and w.

    Args:
        A (np.ndarray) (d,d): linear operator, defining the converged ellipsoid
        w (np.ndarray (d,)): bias, defining the converged ellipsoid
        x (np.ndarray (d,)): input vector

    Returns:
        int: predicted label (+1 or -1)
    """
    return np.sign(np.dot(w, x))

if __name__ == "__main__":
    # uplaod data set
    data_train =        # training set
    data_test =         # test set
    labels_train =   # training labels
    labels_test =    # test labels
    # run the algorithm
    A, w = ellipsoid(data_train, labels_train)

    # predict on test set
    predictions = [predict(w, x) for x in data_test.T]
    # compute error
    error = np.mean(np.not_equal(predictions, labels_test))
    print(f"Test error: {error}")
