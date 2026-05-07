"""
Schwartz-Zippel randomized algorithm for polynomial
identity testing. The algorithm checks if P=0. It is based on the fact that a non-zero polynomial of total degree d
defined over a field F, has at most d roots in F. If we pick a random value for the variable(s) from a set S,
the probability that the polynomial evaluates to zero (and isn't identically zero) is at most d/|S|.

Complexity: For a polynomial of total degree d and T terms the complexity is O(T*d)
"""
import numpy as np

class Polynomial:
    """
    A class to represent a polynomial. For simplicity, we assume that the polynomial is represented as a list of terms,
    where each term is a tuple of the form (coefficient, [exponent1, exponent2, ...]), where the exponents are in the same order as the variables.
    For example, the polynomial 3*x^2*y + 2*x*y^2 + 5 would be represented as [(3, [2, 1]), (2, [1, 2]), (5, [0, 0])].
    """

    def __init__(self, terms):
        self.terms = terms

    def total_degree(self):
        return max(sum(term[1]) for term in self.terms)

    def num_variables(self):
        return len(self.terms[0][1]) if self.terms else 0

    def evaluate(self, values):
        total = 0
        for coeff, exponents in self.terms:
            term_value = coeff
            for i, exp in enumerate(exponents):
                term_value *= (values[i] ** exp)
            total += term_value
        return total

def schwartz_zippel(polynomial: Polynomial) -> bool:
    """
    Returns True if the polynomial is identically zero, and False otherwise.
    """
    d = polynomial.total_degree()
    n = polynomial.num_variables()
    field_size = int(1e4*d)  # a large enough field size to pick from
    S = np.array(range(1, field_size))  # a large enough set of values to pick from
    xs = [np.random.choice(S) for _ in range(polynomial.num_variables())]  # pick random values for the variables
    return polynomial.evaluate(xs) == 0


if __name__ == "__main__":
    # example usage
    poly1 = Polynomial([(3, [2, 1]), (2, [1, 2]), (5, [0, 0])])  # 3*x^2*y + 2*x*y^2 + 5
    poly2 = Polynomial([(0, [2, 1]), (0, [1, 2]), (0, [0, 0])])  # 0
    print(f"Test poly1: {schwartz_zippel(poly1) == False}")  # should return False
    print(f"Test poly2: {schwartz_zippel(poly2) == True}")  # should return True


