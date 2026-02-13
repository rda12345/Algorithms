"""
Conjugate Gradient

The algorithm was originally designed to solve the linear equation A x = b,
where A is symmetric positive definite (SPD) and very large.
The problem is equivalent to minimizing the quadratic function:
f(x) = (1/2) * x.T A x - b.T x
Instead of moving in the steepest descent directions, conjugate gradient moves in
directions that, are mutually A-orthogonal, don't undo previous progress and span
the space efficiently.

Complexity: for an n by n SPD matrix, conjugate_gradient finds the exact solution in at most n iterations.
            Each iteration, requires one matrix vector multiplication.
            In addition, the memory cost is linear, O(n).
"""
import numpy as np
import matplotlib.pyplot as plt


def conjugate_gradient(
        x: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        num_iters: int
)-> np.ndarray:
    history = [x]
    r = b - A @ x
    p = r
    for _ in range(num_iters):
        alpha = np.dot(r, r) / (p.T @ A @ p)
        x_new = x + alpha * p
        r_new = r - alpha * A @ p
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        x = x_new
        r = r_new
        history.append(x)
    return x, history

if __name__ == "__main__":


    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([5, 6, 7])
    num_iters = 30
    x0 = np.array([1, 2, 3])
    x_opt, history = conjugate_gradient(x0, A, b, num_iters)
    error = [np.linalg.norm(A @ v - b) for v in history ]
    print('------------ TEST ------------')

    print(f'Result: {np.all(np.isclose(A @ x_opt, b))}')

    plt.figure()
    plt.plot(range(len(error)), error, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()



