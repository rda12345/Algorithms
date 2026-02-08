"""
QR Decomposition

Given a matrix A of size m x n (with m >= n), a QR
decomposition factorizes it into: A = Q*R,
where Q is an m x n matrix with orthonormal columns (Q^T * Q = I) and
R is an n x n upper triangular matrix.
We can think of the decomposition as breaking A into a rotation (Q) and
scaling (R).


Two algorithms are presented:

1. Decomposition by Gram-Schmidt
2. Decomposition by Given rotations: zero out elements by using plane rotations. Good for
    sparse matrices. Explicitly, the Given rotation G(i,j,a,b) transforms the elements
    a = A[i,k] and b = B[j,k] for a column k, into  r = sqrt(a^2 + b^2)) and 0, correspondingly.
    After a series of rotations we have G_N * G_{N-1} * ... * G_1 * A = R
    Therefore, for  G = G_N * G_{N-1} * ... * G_1, we have A = Q * R, where
    Q = G^T.
    The Given rotations method is numerically stable (orthogonal transformations),
    good for sparse matrices and since each rotation is local it is parallelizable.
"""
import numpy as np

def given_rotation(i:int , j:int, a: float, b: float, m: int) -> np.ndarray:
    """
    Implements a rotation matrix in the plane of coordinates i and j,
    transforming the reduced vector [a, b]^T to [r, 0], with r = sqrt(a^2 + b^2),
    where a = A[i,k] and b = B[j,k].

    Args:
        i (int): row index
        j (int): row index
        a (float): element A[i,k]
        b (float): element A[j,k]
        m (int): number of columns in A

    Returns:
        np.ndarray: rotation matrix
    """
    G = np.eye(m)
    r = np.sqrt(a**2 + b**2)
    c = a/r
    s = b/r
    G[i, i], G[i, j], G[j, i], G[j,j ] = c, s, -s, c
    return G

def QR_given(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a matrix A of size m x n (with m >= n) into Q * R,
    where Q satisfies Q^T * Q = I, and R is upper triangular matrix.

    Args:
        A (np.ndarray): A matrix of size m x n (with m >= n)
    """
    m = A.shape[0]
    G_product = np.eye(m)
    for k in range(A.shape[1]-1):
        for l in range(A.shape[0]-1, k, -1):
            a, b = A[l-1, k], A[l, k]
            G = given_rotation(l-1, l, a, b, m)
            G_product = G @ G_product
            A = G @ A

    return G_product.T, A

def QR_gram_schmidt(A):
    """
    Orthogonalizes the columns of A.

    Args:
        A (np.ndarray): A matrix

    Returns:
        np.ndarray: An orthogonalized matrix, (columns are orthogonal)
    """
    n = A.shape[1]
    V, Q = np.zeros_like(A).astype(float), np.zeros_like(A).astype(float)
    Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
    for k in range(1, n):
        s = 0
        for j in range(k):
            #print(f's: {s}, j: {j}, k: {k}')

            s += np.dot(Q[:,j], A[:, k]) * Q[:,j]
        V[:, k] = A[:, k] - s
        Q[:, k] = V[:, k] / np.linalg.norm(V[:, k])
    return Q, (Q.T @ A)


if __name__ == '__main__':

    print('------------ TESTS ------------')

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 3, 9]])
    Q, R = QR_given(A)
    low_trig = np.tril(R, k = -1)
    print('Given rotations test:')
    print(f'R test: {np.isclose(np.sum(low_trig), 0.0)}')
    print(f'Orthogonality test, Q^T * Q = I: {np.all(np.isclose(Q.T @ Q, np.eye(A.shape[0])))}\n')

    Q, R = QR_gram_schmidt(A)
    low_trig = np.tril(R, k = -1)
    print('Gram-Schmidt test:')
    print(f'R test: {np.isclose(np.sum(low_trig), 0.0)}')
    print(f'Orthogonality test, Q^T * Q = I: {np.all(np.isclose(Q.T @ Q, np.eye(A.shape[0])))}')
