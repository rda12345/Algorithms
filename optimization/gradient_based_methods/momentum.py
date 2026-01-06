"""
Momentum optimization algorithms

The file includes:
- momentum
- Nestrov accelerated gradient (NAG)
- Adagrad
- RMSprop

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
import matplotlib.pyplot as plt

def f(x):
    return (x-1)**2


def momentum(
        objective_function,
        gradient,
        eta: float,
        beta: float,
        x: np.array,
        v: np.array
) -> np.array:
    """
    Evaluates a new search point by momentum algorithm aiming to converge to the
    minimum of the objective function.

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        eta (float): the learning rate, controls the step size
        beta (float): the momentum factor, controls how much of the past
                    gradients are remembered in the current iteration
        x (np.array): initial search point

    Returns:
        float: new search point
    """
    v_new = beta * v + (1 - beta) * gradient(objective_function, x)
    x_new = x - eta * v_new

    return x_new, v_new

def NAG(
        objective_function,
        gradient,
        eta: float,
        beta: float,
        x: np.array,
        v: np.array
) -> np.array:
    """
    Nestrov accelerated gradient (NAG)
    Modifies the standard update rule by calculating the gradient
    at the upcoming position. Considered more efficient due to
    a better understanding of the future trajectory, leading to
    faster convergence, in some cases.
    """
    v_new = beta * v + (1 - beta) * gradient(objective_function, x - eta * v)
    x_new = x - eta * v_new

    return x_new, v_new

# TODO: code adagrad
def adagrad(
        objective_function,
        gradient,
        eta: float,
) -> float:
    pass

# TODO: code RMSprop
def RMSprop(
        objective_function,
) -> float:
    pass





if __name__ == "__main__":

    # momentum
    x, v = np.array([0.0, 2.0]), np.array([1.0, 1.0 ])
    vec = [np.linalg.norm(1 - x)]
    vecNAG = vec.copy()
    for _ in range(100):
        x_new, v_new = momentum(f, gradient, eta=0.1, beta=0.1, x=x, v=v)
        x = x_new
        v = v_new
        vec.append(np.linalg.norm(1-x))

    print(f"Error: {np.linalg.norm(x_new - 1) * 100}%")

    # Nestrov
    x, v = np.array([0.0, 2.0]), np.array([1.0, 1.0 ])
    vec = [np.linalg.norm(1 - x)]
    vecNAG = vec.copy()
    for _ in range(100):
        x_new, v_new = NAG(f, gradient, eta=0.1, beta=0.1, x=x, v=v)
        x = x_new
        v = v_new
        vecNAG.append(np.linalg.norm(1-x))


    plt.figure()
    plt.plot(vec)
    plt.plot(vecNAG)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.legend(["momentum","Nestrov"])
    plt.show()


