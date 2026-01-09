"""
Quasi-Newton Methods (BFGS / L-BFGS)

The algorithms are similar to the Newton's method, but avoids computing
the Hessian directly. Instead, they build an approximation of the Hessian,
or its inverse and update this approximation using the gradient information only.
They are much faster than Newton's method, and tend to converge faster.

- BFGS: approximates the inverse of the Hessian. Widely used, stable and efficient.
- L-BFGS: approximates the Hessian

Complexity: evaluation of the approximation of the inverse of the Hessian scales
            as O(n^2), in contrast to the standard Newton's method, which time-complexity
            scales as O(n^3).
"""
# TODO: fill up the complexity analysis

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

    The learning rate of the optimization algorithm is determined by a line search, which ensures
    the step actually reduces the objective, and that the approximation of the inverse of the
    Hessian remains positive definite.
    The learning rate should satisfy the Wolfe or strong Wolfe conditions.

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

        # choose αk via line search
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
) -> np.ndarray:
    pass

if __name__ == "__main__":
    x = np.array([2.0, 3.0])
    x_opt, history = BFGS(f, gradient, x=x)
    error = [np.linalg.norm(x - x_opt) for x in history]
    print(f"Error: {np.linalg.norm(x_opt) * 100}%")

    #x_opt, history = LBFGS(f, gradient, eta=0.1, x=x)
    #error = [np.linalg.norm(x - x_opt) for x in history]
    #print(f"Error: {np.linalg.norm(x_opt)*100}%")

    plt.figure()
    plt.plot(error)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()
