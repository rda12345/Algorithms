"""
Support Vector Machine

Overview: The SVM is a supervised learning algorithm used for classification of a data set.
The main idea behind SVM is to find a hyperplane that best separates the data points of
different classes in a high-dimensional space.
The SVM can be used for both linear and non-linear classification tasks,
depending on the choice of the kernel function.
The kernel trick allows mapping the data into a high-dimensional feature space and seperating the data in
the high dimensional space efficiently. The method relies on an efficient evaluation of the inner product
between vectors in the feature space. 

Theory: The standard formulation of the SVM, is a solution to the following optimization problem:
find a d-dimensional vector w, such that
<w, x_i> * y_i >= 0 for all i = 1,...,N as possible
or alternatively, find w, such that max_w min_i (sign(<w, x_i>), subject to sign(<w, x_i>) = y_i for all i=1,...,N. 

An alternative formulation of the SVM, is a solution to the following optimization problem:
minimize 1/2 ||w||^2, subject to y_i(<w, x_i>) >= 1 for all i=1,...,N.
Such formulation provides yet another representation of the problem. We utilize the dual formulation (KKT theorem)
to express the problem as an optimization problem over the coefficients {alpha_i}:
maximize sum_i alpha_i - 1/2 sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j),
subject to alpha_i >= 0 for all i=1,...,N, and sum_i alpha_i y_i = 0, 
Here K(x_i, x_j) is nknown as the kernel matrix.
The optimization problem is a quandratic program over the coefficient {alpha_i}. This is solved
by utilizing the built-in cvxpy module (a standard convex optimization package),
which runtime scales as O(N^3) in the number of data points N.

Prediction is achieved by evaluating the sign of sum_i alpha_i y_i K(x_i, x) for a new input vector x,
where the sum is taken over the support vectors, which are the data points with non-zero alpha_i.

We employ the gaussian kernel mapping K(x_i, x_j) = exp(-|x_i - x_j|^2/2*sigma^2),
which maps the data into an infinite-dimensional feature space,
and potentially allows to capture complex non-linear relationships between the data points.


Complexity: The training phase of the SVM involves solving a quadratic program, which has a worst-case complexity of O(N^3).
The prediction phase has a complexity of O(M*N), where M is the number of support vectors and N is the number of data points.
"""
import numpy as np
import cvxpy as cp


def svm(data: np.ndarray, labels: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Returns the weight vector alpha, associated with the SVM algorithm.

    Args:
        data (np.ndarray (N, d)): training data set
        labels (np.ndarray (N,)): training labels
        threshold (float): threshold for considering an alpha value as non-zero

    Returns:
        np.ndarray (N,): the weight vector alpha
    """
    # evaluate the kernel matrix
    N = data.shape[0]
    K = np.eye(N, N)
    for i in range(N):
        for j in range(i + 1, N):
            K[i, j] = kernel(data[i], data[j])
            K[j, i] = K[i, j]       # the kernel matrix is symmetric, so we can save computation by only evaluating the upper triangle
    
    # solve the quadratic program
    alpha = cp.Variable(N)
    K_mat = (labels[:, None] * labels[None, :]) * K  # (N, N) matrix, where K_mat[i, j] = y_i * y_j * K(x_i, x_j)
    objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, cp.psd_wrap(K_mat)))  # psd_wrap ensures that the matrix is treated as positive semidefinite, which bypasses floating point issues that may arise from numerical instability
    constraints = [alpha >= 0, alpha @ labels == 0]
    cp.Problem(objective, constraints).solve()
    alphas = alpha.value

    # filtering out the support vectors (data points with non-zero alpha)
    nonzero_mask = alphas > threshold
    nonzero_idx = np.where(nonzero_mask)[0]
    support_vectors = data[nonzero_idx]
    support_vector_labels = labels[nonzero_idx]
    support_vector_alphas = alphas[nonzero_idx]
    return support_vector_alphas, support_vectors, support_vector_labels

def kernel(x_i, x_j, sigma=1.0):
    """
    Gaussian kernel function.

    Args:
        x_i (np.ndarray): first input vector
        x_j (np.ndarray): second input vector
        sigma (float): bandwidth parameter for the Gaussian kernel

    Returns:
        float: the value of the Gaussian kernel between x_i and x_j
    """
    return np.exp(-np.linalg.norm(x_i - x_j)**2 / (2 * sigma**2))

def predict(data: np.ndarray, support_vectors: np.ndarray, support_vectors_labels: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Evaluates the predictions of the trained SVM machine

    Args:
        data (np.ndarray (N, d)): input data set to predict
        support_vectors (np.ndarray (M, d)): support vectors identified during training
        support_vectors_labels (np.ndarray (M,)): support vector labels
        alphas (np.ndarray (M,)): non-zero weight vector obtained from training

    Returns:
        np.ndarray (N,): predicted labels for the input data set

    """
    # predict each point separately
    def predict_one(x: np.ndarray) -> float:
        kernel_values = np.array([kernel(sv, x) for sv in support_vectors])
        return np.sign(np.sum(alphas * support_vectors_labels * kernel_values))
    return np.array([predict_one(x) for x in data])





if __name__ == "__main__":
    # generate linearly separable 2D data
    np.random.seed(42)
    m = 200

    X_pos = np.random.randn(m // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(m // 2, 2) + np.array([-2, -2])

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(m // 2), -np.ones(m // 2)])
    idx = np.random.permutation(m)
    X, y = X[idx, :], y[idx]

    split = int(0.8 * m)
    data_train, data_test = X[:split, :], X[split:, :]
    labels_train, labels_test = y[:split], y[split:]

    # run the algorithm
    alpha, support_vectors, support_vector_labels = svm(data_train, labels_train)
    predictions = predict(data_test, support_vectors, support_vector_labels, alpha)
    error = np.mean(predictions != labels_test)
    print(f"SVM error: {error:.1f}")

