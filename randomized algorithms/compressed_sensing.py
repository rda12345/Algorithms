"""
Compressed Sensing algorithm (Candes-Romberg-Tao)

Compressed sensing is a technique for efficiently acquiring
and efficiently reconstructing an (approximately) sparse vector (signal) by measuring a small number of measurements 
and solving of a linear system. Approximately sparse means that the vector has
only a few large components. 


Formally, given a k-sparse vector x in R^d, we define the measurement matrix A in R^{m x d},
where m <= d, and the measurement vector y = Ax. Given y, A with the Restricted Isometry Property (RIP)
with parameters k and epsilon, we can reconstruct x by solving the following optimization problem:

    min ||x||_1  subject to Ax = y

The solution determines the original sparse vector x, with high probability.


Restricted Isometry Property (RIP): A matrix A \in R^{m x d} 
has the RIP with parameters k and epsilon if for all k-sparse vectors x in R^d, we have:

    (1 - epsilon) ||x||_2 <= ||Ax||_2 <= (1 + epsilon) ||x||_2

Solution of the LP problem: We rewrite min ||x||_1 as min sum_{i=1}^d t_i subject to -t_i <= x_i <= t_i for all i, and Ax = y,
and then restating thise inequalities in the canonical form required by linear programming solvers.

Matrices with the RIP:
- Random Gaussian matrices: A_ij ~ N(0, 1/m) with m = O(k log(d)) rows.
- Random rows of the DFT matrix: A is a partial Fourier matrix, with m = O(k log^c(d)) rows.
- Best known deterministic constructions: m = O(k^{1.99} log(d)) rows.

Notes:
- The procedure corresponds to a dimension reduction of the (infinite) set of k-sparse vectors in R^d.
- The number of measurements required is m = O(k log(d)/epsilon^2), which is much smaller than d, the original dimension.
- A and x can be complex-valued, but we focus on the real-valued case for simplicity.
- The optimization problem can be solved using linear programming techniques, such as Basis Pursuit or Lasso.

Applications:
- Signal processing (e.g., image and audio compression)
- Magnetic Resonance Imaging (MRI): MRI measurement correspond to a component of the Fourier component
of a sparse image x, Fx, where F is the DiscreteFourier Transform (DFT) matrix.
As a consequence, here the measurement matrix A is a partial Fourier matrix, which has the RIP with high probability.
"""
import numpy as np
import cvxpy as cp

def compressed_sensing(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solves the optimization problem min ||x||_1 subject to Ax = y, where A is the measurement matrix and y is the measurement vector.

    Args:
        A (np.ndarray (m, d)): measurement matrix
        y (np.ndarray (m, )): measurement vector

    Returns:
        np.ndarray (d, ): the reconstructed sparse vector
    """
    d = A.shape[1]
    x = cp.Variable(d)
    prob = cp.Problem(cp.Minimize(cp.norm1(x)), [A @ x == y])
    prob.solve()
    return x.value


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    d = 1000  # original dimension
    k = 10    # sparsity level
    m = 80    # number of measurements
    epsilon = 0.1  # distortion parameter

    # Generate a random k-sparse vector
    x_true = np.zeros(d)
    non_zero_indices = np.random.choice(d, k, replace=False)
    x_true[non_zero_indices] = np.random.randn(k)

    # Generate a random measurement matrix with RIP (A_ij ~ N(0, 1/m))
    A = np.random.randn(m, d) / np.sqrt(m)

    # Generate measurements
    y = A @ x_true

    # Reconstruct the sparse vector
    x_reconstructed = compressed_sensing(A, y)

    # Evaluate the reconstruction error
    error = np.linalg.norm(x_true - x_reconstructed)
    print(f"Reconstruction error: {error:.2e}")
    print(f"log(d)/epsilon^2: {np.log(d)/epsilon**2:.0f}")

