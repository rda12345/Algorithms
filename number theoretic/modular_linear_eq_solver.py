"""
Modular Linear Equation Solver

If there exist a solution, 0 <= x <= n-1,  to the equation a*x = b mod n, the algorithm
returns a list of d = gcd(a,n) solutions.
There exist a solution iff d|b (d divides b).
"""
from euclid import gcd

def modular_linear_eq_solver(a, b, n) -> list | None:
    """
    Modular Linear Equation Solver
    Solves the modular linear equation a*x = b mod n

    Args:
        a (int)
        b (int)
        n (int)

    Returns:
        if a solution exist, list[d], where d = gcd(a,n)
        else None

    """
    d, x, y = gcd(a, n)
    if b % d == 0:
        x0 = (x * (b//d)) % n
        return [(x0 + i * (n//d)) % n for i in range(d)]
    else:
        print("no solutions")


if __name__ == "__main__":
    import numpy as np
    sol1 = modular_linear_eq_solver(a=2, b=34, n=40)
    sol2 = modular_linear_eq_solver(a=3, b=4, n=9)
    print("---------- TESTS ----------")
    print(f'Solution exists test : {np.all(np.isclose(np.array(sol1), np.array([17, 37])))}')
    print(f'Solution does not exist test: {sol2==None}')
