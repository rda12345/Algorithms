"""
Johnson-Lindenstrauss lemma (Gaussian random projection):
For a set of n points in a (high-dimensional) Euclidean space (metric space with l_2 norm),
there exists a mapping to a lower-dimensional space of dimension m = O(log(n)/epsilon^2) such that
the distances between the points are preserved within a factor of (1 ± epsilon) with high probability.

We present two variants of the JL algorithm, which is based on random projections,
the first is defined by normally distributed random variables, while the latter is scheme
involving a random sign matrix.
The guarentees are the same as for the Gaussian random projection, but faster to compute.


Complexity: The algorithm requires O(d*m) time to compute the random projection.
"""
import numpy as np

def johnson_lindenstrauss(X: np.ndarray, m: int) -> np.ndarray:
    """
    Linear random projection, A, of a d-dimensional vector x, on to an m-dimensional space,
    where A_{ij} ~ N(0, 1/m).

    Args:
        X (np.ndarray (d, n)): matrix where each column is a vector in R^{d}
        m (int): dimension of the reduced space

    Return:
        np.ndarray (m, n): reduced space vectors
    """
    d = X.shape[0]
    A = np.random.randn(m, d)/np.sqrt(m)  # random projection matrix, where A_{ij} ~ N(0, 1/m)
    return A @ X  # projected vectors in R^m

def jl_random_sign(X: np.ndarray, m: int) -> np.ndarray:
    """
    Linear random projection, A, of a d-dimensional vector x, on to an m-dimensional space,
    where A_{ij} = +1/sqrt(m) with probability 1/2, -1/sqrt(m) with probability 1/2.

    Args:
        X (np.ndarray (d, n)): matrix where each column is a vector in R^{d}
        m (int): dimension of the reduced space

    Return:
        np.ndarray (m, n): reduced space vectors
    """
    d = X.shape[0]
    A = np.random.choice([1/np.sqrt(m), -1/np.sqrt(m)], size=(m, d), p=[1/2, 1/2])  # random projection matrix
    return A @ X  # projected vectors in R^m


if __name__ == "__main__":
    # generate n random points in d-dimensional space
    np.random.seed(42)
    n = 20  # number of points
    d = 4000  # original dimension
    m = 500  # reduced dimension
    X = np.random.rand(d, n)  # random points in R^d,

    print("---------- Gaussian Variant ---------")
    # apply the JL algorithm
    projected_X = johnson_lindenstrauss(X, m)

    # compute pairwise distances in original and projected space
    original_distances = np.linalg.norm(X[:, None] - X[:, :, None], axis=0)
    projected_distances = np.linalg.norm(projected_X[:, None] - projected_X[:, :, None], axis=0)

    # check distance preservation
    epsilon = 0.1
    preserved = np.all(((1 - epsilon) * original_distances <= projected_distances) & (projected_distances <= (1 + epsilon) * original_distances))
    
    
    print(f"Distance preservation: {preserved}\n")

    print("---------- Random Sign Matrix Variant --------")
    projected_X_achlioptas = jl_random_sign(X, m)
    projected_distances_achlioptas = np.linalg.norm(projected_X_achlioptas[:, None] - projected_X_achlioptas[:, :, None], axis=0)
    preserved_achlioptas = np.all(((1 - epsilon) * original_distances <= projected_distances_achlioptas) & (projected_distances_achlioptas <= (1 + epsilon) * original_distances))
    print(f"Distance preservation: {preserved_achlioptas}")




          
        



