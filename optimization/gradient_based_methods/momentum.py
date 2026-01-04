"""
Momentum algorithm, a.k.a heavy-ball method (Polyak, 1964)

Motivation: standard gradient descent fluctuates in narrow valleys,
and may suffer from slow convergence when gradients point in similar directions.
Momentum fixes these issues by accumulating past gradients.
It accelerates in consistent directions, smooths out oscillations and escapes
shallow minima faster.
"""


import numpy as np
import jax.numpy as jnp
from jax import grad
from gradient_descent import gradient

def f(x):
    return (x-1)**2


def momentum(
        objective_function,
        gradient,
        eta: float,
        beta: float,
        x: float,
        v: float
) -> float:
    """
    Evaluates a new search point by momentum algorithm aiming to converge to the
    minimum of the objective function.

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        eta (float): the learning rate
        beta (float): the momentum factor
        x (np.array): initial search point

    Returns:
        (np.array): new search point
    """
    v_new = beta * v + gradient(objective_function, x)
    x_new = x - eta * v_new

    return x_new, v_new


if __name__ == "__main__":
    x = 0.0
    v = 0.0
    for _ in range(100):
        x_new, v_new = momentum(f, gradient, eta=0.1, beta=0.1, x=x, v=v)
        x = x_new
        v = v_new

    print(f"Error: {np.abs(x_new-1)*100}%")


