"""
Gradient Descent Algorithm

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
import matplotlib.pyplot as plt

def f(x: np.array) -> float:
    """
    Objective function
    """
    return 0.5 * jnp.dot(x, x)




def gradient(f,
              x: np.array,
) -> np.array:
    """
    Evaluates the gradient of f with respect the elements of x.
    Args:
        f (function): the objective function
        x (np.array): initial search point

    Returns:
        np.array: new search point
    """
    return grad(f)(x)


def gradient_descent(
        objective_function,
        gradient,
        eta: float,
        x: float
) -> float:
    """
    Evaluates a new search point by gradient descent, aiming to converge to the
    minimum of the objective function.

    Args:
        objective_function (function): the objective function
        gradient (function): evaluates the gradient of objective function at point x
        eta (float): the learning rate
        x (np.array): initial search point

    Returns:
        np.array: new search point
    """
    return x - eta * gradient(objective_function, x)


if __name__ == "__main__":
    x = np.array([3.0, 2.0])
    vec = [np.linalg.norm([x])]
    for _ in range(100):
        x_new = gradient_descent(f, gradient, eta=0.1, x=x)
        x = x_new
        vec.append(np.linalg.norm(x))

    print(f"Error: {np.linalg.norm(x_new)*100}%")

    plt.figure()
    plt.plot(vec)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()


