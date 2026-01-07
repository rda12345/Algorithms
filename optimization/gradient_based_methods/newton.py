"""
Newton's method

The method utilizes the gradient and the Hessian of the
objective function in the optimization process.
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
        eta: float
) -> float:
    """
    Newton's method

    The method utilizes the gradient and the Hessian of the
    objective function in the optimization process.

    Args:
        x (float): initial search point
        eta (float): learning rate

    Returns:
        float: optimal value
    """
    hessian_f = hessian(f, x)

    x_new = x - eta * np.linalg.inv(hessian(f,x)) @ gradient(f,x)
    return x_new

# TODO: implement newton's method on a function of a vector, where the Hessian is a matrix

if __name__ == "__main__":
    x = np.array([2.0, 3.0])
    vec = [np.linalg.norm(x)]
    for _ in range(100):
        x_new = newton_method(f, gradient, eta=0.1, x=x)
        x = x_new
        vec.append(np.linalg.norm(x))

    print(f"Error: {np.linalg.norm(x_new)*100}%")

    plt.figure()
    plt.plot(vec)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()


