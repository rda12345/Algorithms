"""
Primality testing - Pseudo-primality test and Miller-Rabin primality test

These algorithms are based on Fermat's Little theorem which states that for a prime number, n,
 a^{n-1} = 1 mod n, for all gcd(a,n) = 1.
The inverse of this isn't true, due to Carmichael's numbers, which satisfy a^{n-1} = 1 mod n
for gcd(a,n) = 1 but are composite, e.g., 561 = 3 * 11 * 17.

The strong condition: A a^{n-1} = 1 mod n, for all a = 1,...,n-1 is true (and so is the converse of course).

We present two algorithms:
1. Pseudo-primality test - A primality test which almost works,
    and is good enough for many applications. The algorithm checks if an integer n
    is base-2 pseudo-prime, that is, does it satisfy 2^{n-1} = 1 mod n.
    Composite numbers satisfying this condition are quite rare  (e.g, Carmichael's numbers).

2. Miller-Rabin primality test - Iterative procedure
"""
from modular_exponentiation import modular_exponentiation_iterative
import random

def pseudo_prime(n: int) -> bool:
    """
    Checks if an integer n is base-2 pseudo-prime, that is, does it satisfy 2^{n-1} = 1 mod n.
    Composite numbers satisfying this condition are quite rare (e.g, Carmichael's numbers).

    Args:
        n (int): the integer to be checked

    Returns:
        bool: True if n is base-2 pseudo-prime, False otherwise
    """
    if modular_exponentiation_iterative(2, n-1, n) == 1:
        return True     # definitely
    return False        # probably true



def witness(a: int, n: int) -> bool:
    """
    Returns True if found a witness showing that an integer, n, is composite.
    Is not guaranteed to find such a witness even if n is composite.
    """
    # generating two integers, t and u, satisfying n-1 = 2^t * u
    u = n-1
    t = 0
    while u%2 == 0:  # if n-1 = 2^t, u = 1 while terminate the while loop.
        u = u//2
        t += 1

    x = modular_exponentiation_iterative(a, u, n)
    for i in range(1, t+1):
        x_new = (x**2) % n
        if x_new == 1 and x != 1 and x != n-1:
            return True     # found a non-trivial square-root of 1
        x = x_new
    if x !=1:       #  (a^u)^{2^t} = a^{n-1}, check if n is an a-pseudo-prime.
        return True     # n is composite, since it is not a-pseudo-prime
    return False


def miller_rabin(n: int, s: int) -> bool:
    """
    Primality test for an integer n. Checks if the integer n is base-s pseudo-prime.
    If returns true, the number is definitely composite, otherwise it is almost surely a prime
    (correct with a high probability).

    Args:
        n (int): the integer to be checked
        s (int): number of iterations

    Returns:
        bool: True if n is base-s pseudo-prime, False otherwise
    """
    for _ in range(1, s+1):
        a = random.randint(2,n-2)
        if witness(a, n):
            return False    # definitely
    return True             # almost surely


if __name__ == "__main__":

    print('----------- TESTS -----------')

    n = 123   # large composite
    print(f' pseudo_prime test for a composite number: {pseudo_prime(n = n) == False}')
    print(f' miller_rabin test for a composite number: {miller_rabin(n = n, s = 10) == False}')

    n = 10007    # large prime
    print(f' pseudo_prime test for a prime number: {pseudo_prime(n = n) == True}')
    print(f' miller_rabin test for a prime number: {miller_rabin(n = n, s = 10) == True}')



