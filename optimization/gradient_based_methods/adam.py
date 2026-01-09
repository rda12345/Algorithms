"""
ADAM (Adaptive Momentum Estimation) optimizer

The algorithm works well on complex models and large datasets, using memory
and adapting the learning rate for each parameter automatically.
It builds upon both momentum and RMSprop (root mean square propagation) techniques.
Momentum accelerates the gradient descent by
incorporating a moving average of past gradients (also smooths the optimization trajectory),
while RMSprop is an adaptive learning rate method, using weighted moving average
of squared gradients, which help overcome diminishing learning rates.

"""
# TODO: Complete the file and test

import numpy as np
import jax.numpy as jnp
from jax import grad
from gradient_descent import gradient
import matplotlib.pyplot as plt

def f(x: np.array) -> float:
    """
    Objective function
    """
    return 0.5 * jnp.dot(x, x)



def adam(
        objective_function,
        gradient,
        beta: float,
        xi: float,
        eta: float,
        x: np.ndarray,
        v: np.ndarray,
        max_iter: int = 100,
        threshold: float = 1e-6,
) -> float:
    """

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        beta (float): momentum factor
        xi (float):
        eta (float): learning rate
        x (np.array): initial search point
        v (np.array): initial second momentum
        max_iter (int): maximum number of iterations
        threshold (float): convergence threshold

    Returns:
        tuple (np.ndarray, list[np.ndarray]): new search point, history
    """
    history = [x]
    m = np.zeros_like(x)
    eps = 1e-6
    for t in range(1,max_iter + 1):
        g = gradient(objective_function, x)
        m_new = beta * m + (1 - beta) * g
        v_new = xi * v + (1 - xi) * g**2


        # update baised moment estimates
        m_hat = m_new / (1-beta**t)
        v_hat = v_new / (1-xi**t)

        # parameter update
        x_new = x - eta * m_hat / (np.sqrt(v_hat) + eps)

        # storing the new search point in the history
        history.append(x_new)

        if np.linalg.norm(x_new-x) < threshold:
            return x_new, history

        x = x_new
        v = v_new
        m = m_new


    return x_new, history


if __name__ == "__main__":
    x = np.array([3.0, 2.0])
    v = np.array([0.0, 0.0])
    x_opt, history = adam(f, gradient, beta=0.9, xi=0.9, eta=0.1, x=x, v=v, max_iter=200)

    error = [np.linalg.norm(x) for x in history]
    print(error)
    print(f"Error: {np.linalg.norm(x_opt)*100}%")

    plt.figure()
    plt.plot(error)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()



