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
import random

def f(x: np.array) -> float:
    """
    Objective function
    """
    return 0.5 * jnp.dot(x, x)


def momentum(
        objective_function,
        gradient,
        eta: float,
        beta: float,
        x: np.array,
        v: np.array,
        max_iter: int = 100,
        threshold: float = 1e-6
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
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold

    Returns:
        np.ndarray: new search point
    """
    history = []
    for i in range(max_iter):
        v_new = beta * v + (1 - beta) * gradient(objective_function, x)
        x_new = x - eta * v_new
        history.append(x_new)
        if np.linalg.norm(x_new-x) < threshold:
            return x_new, history
        x = x_new
        v = v_new
    return x_new, history

def NAG(
        objective_function,
        gradient,
        eta: float,
        beta: float,
        x: np.array,
        v: np.array,
        max_iter: int = 100,
        threshold: float = 1e-6
) -> np.array:
    """
    Nestrov accelerated gradient (NAG)
    Modifies the standard update rule by calculating the gradient
    at the upcoming position. Considered more efficient due to
    a better understanding of the future trajectory, leading to
    faster convergence, in some cases.
    """
    history = []
    for _ in range(max_iter):
        v_new = beta * v + (1 - beta) * gradient(objective_function, x - eta * v)
        x_new = x - eta * v_new
        history.append(x_new)
        if np.linalg.norm(x_new-x) < threshold:
            return x_new, history
        x = x_new
        v = v_new
    return x_new, history


def adagrad(
        objective_function,
        gradient,
        G: float,
        eta: float,
        eps: float,
        x: np.array,
        max_iter: int = 100,
        threshold: float = 1e-6
) -> float:
    """
    Adagrad optimization algorithm. Adjusts the learning
    rate of each parameter during the learning process.

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        G (float): sum of squared gradients
        eta (float): global learning rate
        eps (float): small value added to avoid divergence of the learning rate
        x (np.array): search point
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold

    Returns:
        np.ndarray: new search point
    """
    history = []
    for i in range(max_iter):
        if np.linalg.norm(G) < eps:
            learning_rate = eta
        else:
            learning_rate = eta / np.sqrt(G + eps)
        g = gradient(objective_function, x)
        x_new = x - learning_rate * g
        G += g**2
        history.append(x_new)
        if np.linalg.norm(x_new - x) < threshold:
            return x_new, history
        x = x_new

    return x_new, history



# TODO: code RMSprop
def RMSprop(
        objective_function,
        gradient,
        G: float,
        eta: float,
        eps: float,
        beta: float,
        x: np.array,
        max_iter: int = 100,
        threshold: float = 1e-6
) -> float:
    """
    RMSprop optimization algorithm. Adjusts the learning rate
    of each parameter during the learning process using a moving
    average of the squared gradients.

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        G (float): moving average of squared gradients
        eta (float): global learning rate
        eps (float): small value added to avoid divergence of the learning rate
        beta (float): controls the memory of the moving average
        x (np.array): search point
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold


    Returns:
        np.ndarray: new search point
    """
    history = []
    for i in range(max_iter):
        if np.linalg.norm(G) < eps:
            learning_rate = eta
        else:
            learning_rate = eta / np.sqrt(G + eps)
        g = gradient(objective_function, x)
        x_new = x - learning_rate * g
        G = beta * G + (1 - beta) * g**2
        history.append(x_new)
        if np.linalg.norm(x_new - x) < threshold:
            return x_new, history
        x = x_new
    return x_new, history


if __name__ == "__main__":
    x, v = np.array([0.0, 2.0]), np.array([0.0, 0.0 ])
    vec = [np.linalg.norm(x)]
    vecNAG = vec.copy()
    vecAdagrad = vec.copy()
    vecRMSprop = vec.copy()

    # momentum

    x_opt, history = momentum(f, gradient, eta=0.1, beta=0.1, x=x, v=v)
    error_momentum = [np.linalg.norm(x) for x in history]

    # Nestrov
    x_opt, history= NAG(f, gradient, eta=0.1, beta=0.1, x=x, v=v)
    error_NAG = [np.linalg.norm(x) for x in history]

    # Adagrad
    G = 0
    eta = 0.1
    eps = 1e-8
    x_opt, history = adagrad(f, gradient, G, eta, eps, x)
    error_adagrad = [np.linalg.norm(x) for x in history]


    # RMSprop
    beta = 0.9
    x_opt, history = RMSprop(f, gradient, G, eta, eps, beta, x)
    error_RMSprop = [np.linalg.norm(x) for x in history]

    # Plot results
    plt.figure()
    plt.plot(error_momentum)
    plt.plot(error_NAG)
    plt.plot(error_adagrad)
    plt.plot(error_RMSprop)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend(["Momentum","Nestrov","Adagrad","RMSprop"])
    plt.show()


