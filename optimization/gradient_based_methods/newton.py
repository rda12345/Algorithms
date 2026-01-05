"""
Newton's method

The method utilizes the gradient and the Hessian of the
objective function in the optimization process.
"""
import numpy as np
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt


from optimization.gradient_based_methods.gradient_descent import gradient

def f(x):
    return (x-1)**2

def hessian(f,x):
    df_dx2 = grad(grad(f))
    return df_dx2(x)

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
    x_new = x - eta * (1/hessian(f,x)) * gradient(f,x)
    return x_new

# TODO: implement newton's method on a function of a vector, where the Hessian is a matrix

if __name__ == "__main__":
    x = 0.0
    vec = [1-x]
    for _ in range(100):
        x_new = newton_method(f, gradient, eta=0.1, x=x)
        x = x_new
        vec.append(np.abs(1-x))

    print(f"Error: {np.abs(x_new-1)*100}%")

    plt.figure()
    plt.plot(vec)
    plt.xlabel("Iteration")
    plt.ylabel("error")
    plt.show()
