"""
Pollard Rho Algorithm

A heuristic algorithm to find the factors of a composite number.


Complexity: Since it is a heuristic algorithm there are no guarantees regarding it running time
            or success. Nevertheless, we expect that the algorithm finds a prime in O(sqrt(p)) iterations,
            where p is a prime factor of n. Therefore, the expected running time is O(n^{1/4}).
            The algorithm requires only a constant number of memory allocations.
"""
import random
from euclid import gcd
import numpy as np



def pollard_rho(n: int) -> list[int]:
    """
    Heuristic algorithm to find the factors of a composite number.

    Args:
        n (int): composite number

    Returns:
        list[int]: factors
    """
    factors = set()
    x = random.randint(0,n-1)
    y = x
    k = 2
    constant = 100
    for i in range(constant*int(np.ceil(n**(1/4)))):  # the factor constant is chosen arbitrarily, the important part is the scaling
        x = (x**2 - 1) % n
        d, _, _ = gcd(y - x, n)
        if d != 1 and d != n:
            if d not in factors:
                factors.add(d)
        if i == k:
            y = x
            k = 2*k
    return factors

if __name__ == "__main__":

    n = 13*5*3
    factors = pollard_rho(n)
    print(f'factors: {factors}')


