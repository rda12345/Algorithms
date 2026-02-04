"""
Modular Exponentiation

Two algorithms, one recursive, and one iterative, are presented. They perform
modular exponentiation of an integer a.
The algorithm is based on the relation

a^b = (a^{b/2})^2 if b mod(2) == 0, a*a^{b-1} if b is odd, and 1 otherwise.

Complexity: The algorithm performs O(log(n)) operations, assuming a and b are
            are lg(n)-bit numbers.
"""

def modular_exponentiation(a, b, n):
    """
    Modular Exponentiation

    Args:
        a (int)
        b (int): exponent
        n: modulus

    Returns:
        int: modular exponentiation
    """
    if b == 0:
        return 1
    if b % 2 == 0:
        d = modular_exponentiation(a, b//2, n)
        return (d * d) % n
    else:
        d = modular_exponentiation(a, b-1, n)
        return (a * d) % n

def modular_exponentiation_iterative(a, b, n):
    d = 1
    if b == 0:
        return d
    temp = a
    while b > 0:
        if b & 1 == 1:      # the most least significant bit of the current b is 1
            d *= temp
            d %= n
        temp *= temp
        b >>= 1
    return d


if __name__ == '__main__':
    print('----------- TESTS -----------')
    a = 1359
    b = 2678
    n = 3491
    print(f'Test recursive algorithm: {modular_exponentiation(a, b, n) == (a**b) % n}')

    a = 7249
    b = 2783
    n = 5293
    print(f'Test iterative algorithm: {modular_exponentiation_iterative(a, b, n) == (a**b) % n}')
