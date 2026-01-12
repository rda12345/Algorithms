"""
Quasi-Newton Methods (BFGS / L-BFGS)

The algorithms are similar to the Newton's method, but avoids computing
the Hessian directly. Instead, they build an approximation of the Hessian,
or its inverse and update this approximation using the gradient information only.
They are much faster than Newton's method, and tend to converge faster.

- BFGS: Approximates the inverse of the Hessian. Widely used, stable and efficient.

Key idea: The curvature of a function (encoded by the Hessian) can be approximated
    by the change of the gradient. For a quadratic function the gradient change:
    y_k = \nabla f(x_{k+1}) - \nabla f(x_{k}) = \nabla^2 f(x_{k}) s_k, where
    s_k = x_{k+1} - x_{k}.
    Therefore, the inverse of the Hessian at iteration H_{k+1} satisfies teh secant equation
    H_{k+1} y_k = s_k. The algorithm enforces the secant equation, thus, enforcing
    that the Hessian estimate behaves correctly along the direction moved (satisfies
    the curvature constraint).
    The update rule of the Hessian (is constructed in such a way such that it) satisfies the
    secant condition, is symmetric and positive definite.
    This leads to numerical stability and efficiency of the algorithm.

Note: Many implementations of the BFGS determine the learning rate by a line search.
    This improves the numerical stability and efficiency of the algorithm. The present
    implementation assumes a small constant learning rate.

- L-BFGS (Limited memory BFGS): Doesn't store the approximation of the inverse
        Hessian, but only the last m curvature pairs. From the m pairs
        the algorithm evaluates an inverse Hessian which equivalent to the
        BFGS inverse Hessian restricted to the last m updates.

Complexity: BFGS: Evaluation of the approximation of the inverse of the Hessian scales
            as O(n^2), in contrast to the standard Newton's method, which time-complexity
            scales as O(n^3) (evaluating the inverse of a matrix scales as
            matrix multiplication, so the n^3 can be improved applying divide and
            conquer algorithm, such as the Strassen algorithm.).

            Memory complexity of storing the approximation of the inverse of the Hessian,
            H_{k}, scales as O(n^2).

            For large n (e.g. machine learning applications) this becomes
            infeasible.

            L-BFGS only stores the last m curvature pairs (s_k, y_k). Therefore, the space
            complexity of L-BFGS is O(m*n).
"""
import queue
from collections import deque

import numpy as np
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from gradient_descent import gradient



def f(x: np.array) -> float:
    """
    Objective function
    """
    return 0.5 * jnp.dot(x, x)


def BFGS_inv_hessian(
        rho: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        H: np.ndarray
)-> np.array:
    """
    Approximates the inverse of the Hessian, using the secant condition.
    """
    return (np.eye(len(s)) - rho * np.outer(s, y)) @ H @ (np.eye(len(s)) - rho * np.outer(y, s)) + rho * np.outer(s, s)

def LBFGS_gradient_direction(
        storage: list[tuple[np.ndarray, np.ndarray, float]],
        g: np.ndarray,
        m: int = 10,
) -> np.ndarray:
    q = g
    alpha_list = []
    for i in reversed(range(m)):
        s, y, rho = storage[i]
        alpha_i =  rho * np.outer(s, y)
        alpha_list.append(alpha_i)
        q -= alpha_i @ y

    alpha_list = reversed(alpha_list)

    s, y, _ = storage[m]
    gamma = np.outer(s, y) / np.dot(y, y)
    H0 = gamma * np.eye(len(s))
    r = H0 @ q

    for i in range(m):
        s, y, rho = storage[i]
        beta = rho * np.outer(y, r)
        r += (a[i] - beta) * s

    return r




def BFGS(
        objective_function,
        gradient: np.ndarray,
        x: np.ndarray,
        eta: float = 0.1,
        max_iter: int = 100,
        threshold: float = 1e-6
) -> float:
    """
    BFGS quasi-Newton method

    The method utilizes the gradient and an approximation of the inverse of the Hessian to optimize the
    objective function.

    Args:
        objective_function
        gradient (np.ndarray): gradient of the objective function at point x
        x (np.ndarray): initial search point
        eta (float): learning rate
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold

    Returns:
        float: optimal value
    """
    history = []
    H = np.eye(len(x))  # initialization of the approximation of the inverse Hessian
    for _ in range(max_iter):
        g = gradient(objective_function, x)

        p = H @ g
        x_new = x - eta * p
        history.append(x_new)

        s = x_new - x
        y = gradient(objective_function, x_new) - g
        ys = np.dot(y, s)
        rho = 1.0 / ys

        # convergence check
        if np.linalg.norm(x_new - x) < threshold or ys <= 1e-10:
            return x_new, history

        H = BFGS_inv_hessian(rho, s, y, H)
        x = x_new

    return x_new, history


# TODO: code LBFGS

def LBFGS(
        objective_function,
        gradient: np.ndarray,
        x: np.ndarray,
        m:int = 10,
        eta: float = 0.1,
        max_iter: int = 100,
        threshold: float = 1e-6
) -> float:
    """
    L-BFGS (Limited memory BFGS) quasi-Newton method

    The method utilizes the gradient and an approximation of the inverse of the Hessian to optimize the
    objective function. In contrast to the BFGS, L-BFGS does not store an approximation of the
    inverse of the Hessian, but evaluates it on the fly, utilizing m latest curvature pairs.

    Args:
        objective_function
        gradient (np.ndarray): gradient of the objective function at point x
        x (np.ndarray): initial search point
        m (int): maximum number of curvature pairs forming the inverse Hessian approximation
        eta (float): learning rate
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold

    Returns:
        float: optimal value
    """
    history = []
    storage = []
    q = queue.PriorityQueue()
    H = np.eye(len(x))  # initialization of the approximation of the inverse Hessian
    g = gradient(objective_function, x)
    p = H @ g
    for _ in range(max_iter):

        x_new = x - eta * p
        history.append(x_new)

        s = x_new - x
        y = gradient(objective_function, x_new) - g
        ys = np.dot(y, s)
        rho = 1.0 / ys

        if np.linalg.norm(x_new - x) < threshold or ys <= 1e-10:
            return x_new, history
        x = x_new

        # updating the queue
        if len(q) < m:
            q.push((s, y, rho))
        else:
            q.pop()
            q.push((s, y, rho))

        # evaluating the gradient direction
        p = LBFGS_search_direction(storage=q.tolist(), g=g, m=m)


    return x_new, history

if __name__ == "__main__":
    x = np.array([2.0, 3.0])
    #x_opt, history = BFGS(f, gradient, x=x)
    #error = [np.linalg.norm(x - x_opt) for x in history]
    #print(f"Error: {np.linalg.norm(x_opt) * 100}%")

    x_opt, history = LBFGS(f, gradient, eta=0.1, x=x)
    error = [np.linalg.norm(x - x_opt) for x in history]
    print(f"Error: {np.linalg.norm(x_opt)*100}%")

    plt.figure()
    plt.plot(error)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()
