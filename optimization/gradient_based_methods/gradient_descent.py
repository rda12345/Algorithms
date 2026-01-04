"""
An implementation of the gradient descent algorithm

The algorithm evaluates the gradient of the objective function and performs
a step in the direction opposite to the gradient.

For a differentiable, convex L-smooth objective function, for a sufficiently small
learning rate, eta, the algorithm converges to the global minimum.
The condition on the learning rate is 0 < eta < 2/L, where L is the
Lipschitz constant.
"""
import numpy as np
import jax.numpy as jnp
from jax import grad

def f(x):
    return (x-1)**2


def gradient(f, x):
    df_dx = grad(f)
    return df_dx(x)


def gradient_descent(objective_function, gradient, eta: float, x: np.array)\
        -> np.array:
    """
    Evaluates a new search point by gradient descent, aiming to converge to the
    minimum of the objective function.

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        x (np.array): initial search point

    Returns:
        (np.array): new search point
    """
    grad_obj = gradient(objective_function, x)
    x_new = x - eta * grad_obj
    return x_new


if __name__ == "__main__":
    x = 0.0
    for _ in range(100):
        x_new = gradient_descent(f, gradient, eta=0.1, x=x)
        x = x_new
        print(x_new)

