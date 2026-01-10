"""
Newton's method

The method utilizes the gradient and the Hessian of the
objective function in the optimization process.

Complexity: Evaluation of the inverse of the Hessian scales (naively) as O(n^3)
            at each iteration using the LU or LUP decompositions to invert the
            Hessian. However, asymptotically, evaluating the inverse of a matrix
            scales as matrix multiplication, so the n^3 can be improved by
            applying the divide and conquer algorithm such as Strassen algorithm.
            So the asymptotic complexity is approximately O(n^{2.37}).
"""
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd
import matplotlib.pyplot as plt
from optimization.gradient_based_methods.gradient_descent import gradient

def f(x: np.array) -> float:
    """
    Objective function
    """
    return 0.5 * jnp.dot(x, x)

def hessian(f,x):
    return jacfwd(grad(f))(x)

def newton_method(
        objective_function,
        gradient,
        x: float,
        eta: float,
        max_iter: int = 100,
        threshold: float = 1e-6
) -> float:
    """
    Newton's method

    The method utilizes the gradient and the Hessian of the
    objective function in the optimization process.

    Args:
        x (float): initial search point
        eta (float): learning rate
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold

    Returns:
        float: optimal value
    """
    history = []
    for _ in range(max_iter):
        x_new = x - eta * np.linalg.inv(hessian(f,x)) @ gradient(f,x)
        history.append(x_new)
        if np.linalg.norm(x_new - x) < threshold:
            return x_new, history
        x = x_new
    return x_new, history


if __name__ == "__main__":
    x = np.array([2.0, 3.0])
    x_opt, history = newton_method(f, gradient, eta=0.1, x=x)
    error = [np.linalg.norm(x - x_opt) for x in history]
    print(f"Error: {np.linalg.norm(x_opt)*100}%")

    plt.figure()
    plt.plot(error)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()


