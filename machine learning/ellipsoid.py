"""
Ellipsoid learner algorithm

An online learning algorithm which maintains an ellipsoid that contains the V_t, the set of hypotheses
which are correct on all the t first examples x_1,...,x_{t-1}.
Assumes realizability, i.e., there is a linear separator in the set of hypotheses,
and that the data is linearly separable.
Each mistake reduces the volume of the ellipsoid, E_t, by a factor e^{-1/(2*d+2)}.
Moreover, the volume of E_t is bounded by Vol(B) * (1/n)^{2*d}, (E_t contains a ball of radius 1/n^{2*d}
around the optimal hypothesis, w*).
Therefore it converges rapidly to the best hypothesis.

The algorithm is guaranteed to converge after at most (2*d+2)*log(R) mistakes,
where R is the radius of the smallest ball containing the set of hypotheses.
Complexity:
    
"""
import numpy as np
import matplotlib.pyplot as plt

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
    alpha = (d**2 / (d**2-1))**(1/2)
    for t in range(N):    
        x = data[:,t]
        y = labels[t]
        y_pred = np.sign(np.dot(w, x))   # prediction of y
        if y_pred != y:
            temp = A@x/(np.sqrt(x.T @ A @ x))
            w += (y / (d+1)) * temp 
            A =  alpha * (A - (2/(d+1))*(np.outer(temp, temp)))
    return A, w

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
    d = 2
    n_point = 500
    data_train = np.random.randn(d, n_point)
    data_test  = np.random.randn(d, n_point)
    # true separator: w* = [1, 0], hyperplane x[0] = 0 (passes through origin)
    labels_train = np.sign(data_train[0, :]).astype(int)
    labels_test  = np.sign(data_test[0, :]).astype(int)

    # run the algorithm
    A, w = ellipsoid(data_train, labels_train)

    # predict on test set
    predictions = np.array([predict(w, x) for x in data_test.T])
    error = np.mean(predictions != labels_test)
    print(f"Learned w: {w}")
    print(f"Test error: {error:.4f}")


    # plot train and test with label colors
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, labels, title in [
        (axes[0], data_train, labels_train, "Train"),
        (axes[1], data_test,  labels_test,  "Test"),
    ]:
        for label, color in [(1, "blue"), (-1, "red")]:
            mask = labels == label
            ax.scatter(data[0, mask], data[1, mask], c=color, label=f"y={label:+d}", alpha=0.4, s=10)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()
