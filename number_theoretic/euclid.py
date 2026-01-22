"""
Euclid's and Extended Euclid's algorithms

Both algorithms compute the greatest common divisor (gcd) between two non-negative integers,
utilizing the identity gcd(a,b) = gcd(b, a mod(b))
Extended Euclid's algorithm also returns the (possibly negative) integer
coefficients x, y, such that gcd(a,b) = a*x + b*y.

Complexity: O(lg(b)). The overall runtime is proportional to the number of recursive calls.
            Utilizing Lame theorem, which states that for an integer k>=1, if a>1 and b>=1, and
            b<F_{k+1}, where F_k is the k'th Fibonacci number, then the call euclid(a,b) makes fewer than k
            recursive calls (proven by showing that euclid(F_{k+1},F_k) makes k calls.
            Since F_k \approx phi^k/\sqrt{5}, where \phi is the golden ratio, we get a logarithmic
            number of recursive calls (also scales as the number of arithmetic operations).
            Assuming bit multiplication and division takes O(lg(b)^2) operations (can be improved using Karatsuba algorithm).
            euclid(a,b) performs O(lg(b)^3) bit operations.
"""

def euclid(a,b):
    """
    Evaluates the greatest common divisor of two  non-negative integers.
    """
    if b == 0:
        return a
    return euclid(b,a%b)


def gcd(a,b):
    """
    Evaluates the greatest common divisor of two integers, employing the extended Euclid's algorithm.
    Args:
        a (int): non-negative integer
        b (int): non-negative integer

    Returns:
        tuple[int]: greatest common divisor of a and b, and (possibly negative) x, y,
                    integers, satisfying gcd(a,b) = a*x + b*y.
    """
    if b == 0:
        return (a,1,0)
    (d, x, y) = gcd(b, a % b)
    return (d, y, x - (a // b) * y)


if __name__ == "__main__":
    print('----------- TESTS -----------')
    print('Euclid\'s algorithm')
    print(f'Primes: {euclid(2,5) == 1}')
    print(f'b=0: {euclid(3,0) == 3}')
    print(f'a=0: {euclid(0, 10) == 10}')
    print(f'Composite numbers: {euclid(30, 24) == 6}')

    print('Extended Euclid\'s algorithm')
    g, x, y = gcd(a=11, b=9)
    print(f'Primes: {g == 1 &  11*x + 9*y == 1}')
    g, x, y = gcd(a=10, b=0)
    print(f'b=0: {g == 10 & 10*x == 10}')
    g, x, y = gcd(a=0, b=10)
    print(f'a=0: {g == 10 & 10 * y == 10}')
    g, x, y = gcd(a=30, b=24)
    print(f'Composite numbers=0: {g == 6 & 30 * x + 24 * y == 6}')




