"""
Simplex Method

Algorithm to solve linear programing (LP) optimization problems.
An LP is an optimization problem where the optimization function and
constraints are linear equations/inequalities:

maximize_{x_1,...,x_n}        \sum_i c_i x_i,
subject to \{\sum_j a_{ij} x_j <= b_i\}
            x_1,...,x_n >= 0.

slack form: convert the first kind of inequalities above to equalities
    by introducing slack variables.

    x_{n+i} = b_j - \sum_j a_{ij} x_j
    x_{n+i} > 0

    Since the original inequality is true only if and only if the second
    equality/inequality are true, the original and slack formulations of the
    LP are equivalent.

    In the slack form, the variables on the l.h.s are called basic variables,
    and the those on the r.h.s of the equalities are called non-basic variables.
    We also express the optimization function as z =\nu + \sum_i c_i x_i, where
    \nu is a constant (doesn't influence the optimization).

    As the algorithm proceeds the basic and non-basic variables change, but
    the number of basic/non-basic variables does not change.

The algorithm: The basic operation is a "pivot", exchanging a basic and
non-basic variables.
This is achieved by finding a basic feasible solution (a feasible solution, where
all the non-basic variables are set to zero. Choose a variable in the optimization
function which coefficient is positive, this is the "entering" variable
and increase as much as possible, without violating the constraints
(the basic variables must be non-negative).
The leaving variable is the basic variable which limits the increase of
the entering variable.

The algorithm performs repeated pivots until all the coefficients of the
optimization function are negative. Setting the variables in the optimization
function to zero, gives the maximum value and the value of the basic variables
is the optimal solution.

Complexity: Exponential in the number of variables, constraints but
typically can be solved in polynomial for standard cases.
"""
from cmath import inf

import numpy as np
from numpy.ma.core import nonzero


def pivot(
        N: list,
        B: list,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        nu: float,
        l: int,
        e: int,
) -> tuple:
    """
    Performs a pivot in the simplex method.
    Assumes the LP is already in slack form.

    Args:
        N (set): contains the indices of the non-basic variables.
        B (set): contains the indices of the basic variables.
        A (np.ndarray): contains the coefficients of the constraints.
        b (np.ndarray): contains the bounds
        c (np.ndarray): optimization coefficients
        nu (float): optimization function constant
        l (int): leaving variable (a basic variable)
        e (int): entering variable (a non-basic variable)
    """
    A_new = np.zeros_like(A)
    b_new = np.zeros_like(b)
    c_new = np.zeros_like(c)

    # Compute the coefficients of the equation for a new basic variable
    b_new[e] = b[l]/A[l,e]
    for j in N:
        if j != e:
            A_new[e,j] = A[l,j] / A[l,e]
    A_new[e,l] = 1 / A[l,e]  # l is in B not in N so it is not included in the sum

    # Compute the coefficients of the remaining constraints
    for i in B:
        if i != l:
            b_new[i] = b[i] - A[i,e] * b_new[e]
            for j in N:
                if j != e:
                    A_new[i,j] = A[i,j] - A[i,e] * A_new[e,j]
            A_new[i,l] = - A[i,e] * A_new[e,l]

    # Compute the objective function
    nu_new = nu + c[e] * b_new[e]
    for j in N:
        if j != e:
            res = c[j] - c[e] * A_new[e,j]
            c_new[j] = res
    c_new[l] = - c[e] * A_new[e,l]

    # Compute the new sets of basic and non-basic variables
    N_new = (N - {e}) | {l}
    B_new = (B - {l}) | {e}

    return N_new, B_new, A_new, b_new, c_new, nu_new

#TODO: code the initialize_simplex properly
def initialize_simplex(
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
) -> tuple:
    """
    Initializes the simplex method.
    Checks if the program is infeasible, if so it terminates, if not returns a
    slack form for which the initial basic solution is feasible.



    """
    n = len(A)
    m = len(nonzero(b)[0])
    N = {i for i in range(n-m)}
    B = {i + m for i in range(n-m)}
    nu = 0
    return N, B, A, b, c, nu



def simplex(
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
) -> np.ndarray:
    """
    Simplex Method

    Solves a Linear Program in slack form.
    Chooses the entering and leaving variables deterministically, according
    to a maximum/minimum rule.

    Args:
        A (np.ndarray): coefficients of the constraints.
        b (np.ndarray): bounds of the constraints.
        c (np.ndarray): optimization coefficients

    Returns:
        tuple  (np.ndarray, float): solution of the linear program
                        and optimal (maximal) value.
    """
    N, B, A, b, c, nu = initialize_simplex(A, b, c)
    n = len(b)
    Delta = np.zeros(n)
    # from j in N choose j, such that c_j > 0
    while np.any(c[list(N)] > 0):
        # choose the maximum coefficient in the
        e = np.argmax(c)
        for i in B:
            if A[i,e] > 0:
                Delta[i] = b[i] / A[i,e]
            else:
                Delta[e] = inf
        nonzero_indices = np.where(Delta > np.min(Delta))[0]
        l = nonzero_indices[np.argmin(Delta[nonzero_indices])]   # choosing the index for which Delta > 0 and minimum.
        if Delta[l] == inf:
            return 'unbounded'
        else:
            N, B, A, b, c, nu = pivot(N, B, A, b, c, nu, l, e)  #perform a pivot
    x = np.zeros(n)  # initializing the solution array
    for i in range(n):
        if i in B:
            x[i] = b[i]
    return x, nu


if __name__ == "__main__":

    N = {0,1,2}
    B = {3,4,5}
    n = 6
    temp = np.array([[1, 1, 3], [2, 2, 5], [4, 1, 2]]).astype(float)
    A = np.zeros((n,n)).astype(float)
    A[3:,:3] = temp
    b = np.array([ 0, 0, 0, 30, 24, 36]).astype(float)
    c = np.array([3, 1, 2, 0, 0, 0]).astype(float)


    # converting the array elements to floats
    A = A
    nu = 0.0
    l = 5
    e = 0

    # Pivot test
    #N_new, B_new, A_new, b_new, c_new, nu = pivot(N=N, B=B, A=A, b=b, c=c, nu=nu, l=l, e=e)

    # Simplex test
    solution, optimal_value = simplex(A, b, c)
    print("#------------- TESTS -------------")
    print(f'Optimal solution: {list(solution) == [8. , 4., 0. , 18., 0., 0.]}')
    print(f'Optimal value: {optimal_value == 28.0}')