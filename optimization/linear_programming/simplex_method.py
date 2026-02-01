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

If the initial input is a feasible solution (satisfies all the constraints,
with non-negative variables (x_i >= 0) and non-negative constraint bounds (b_i >= 0)),
the algorithm is guaranteed to return either a feasible optimal solution
or report that the solution is unbounded.


The algorithm performs repeated pivots until all the coefficients of the
optimization function are negative. Setting the variables in the optimization
function to zero, gives the maximum value and the value of the basic variables
is the optimal solution.

Complexity: Exponential in the number of variables, constraints but
typically can be solved in polynomial for standard cases.
"""
from cmath import inf, isclose

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


def initialize_simplex(
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
) -> tuple:
    """
    Initializes the simplex method.
    Checks if the program is infeasible, if so it terminates, if not returns a
    slack form for which the initial basic solution is feasible.
    A feasible solution satisfies all the linear program constraints.

    Args:
        A (np.ndarray): coefficients of the constraints.
        b (np.ndarray): bounds of the constraints.
        c (np.ndarray): optimization coefficients

    Returns:
        tuple: containing the non-basic set of indices, basic set of indices, arrays A, b, c and a float nu
    """
    num_var = len(A)
    b_nonzeros = b.nonzero()[0]
    B = {int(i) for i in b_nonzeros}
    N = {int(i) for i in range(num_var)}.difference(B)
    n = len(b_nonzeros) # number of basic variables
    k = int(np.argmin(b))
    nu = 0
    # checks if the input provides a feasible solution
    if b[k] >= 0:
        return N, B, A, b, c, nu
    # if the initial solution isn't feasible then it won't be a feasible solution for the auxiliary LP.
    # form an auxiliary LP
    c_original = np.concatenate((np.zeros(1).astype(float),c.copy()), axis=0)
    col = np.zeros_like(b)
    col[b_nonzeros] = -np.ones_like(b_nonzeros)
    row = np.zeros(5).astype(float)
    A_aux = np.row_stack((row,np.column_stack((col,A)))) # addition of -\bar{x}_0 (+1 in the slack form) to the left
                                                           # hand side of all the constraints and a new row

    # Definition the auxiliary variables (shifting the indices by one)
    N_aux = {i + 1 for i in N}.union({0})
    B_aux = {i + 1 for i in B}
    b_aux = np.zeros(num_var+1).astype(float)
    b_aux[1:] = b
    c_aux = np.zeros(num_var+1).astype(float)
    c_aux[0] = -1
    l = k+1     # the index of the leaving basic variable (corresponding to the variable which the most negative element in b)
    (N, B, A, b, c, nu) = pivot(N_aux, B_aux, A_aux, b_aux, c=c_aux, nu=0, l=l, e=0)     # performs a pivot, where x_0 and x_l are the entering and leaving variables
    # The basic solution is now feasible for L_aux

    Delta = np.zeros(num_var+1)
    # solve the auxiliary LP
    while np.any(c[list(N)] > 0):
        # choose the maximum coefficient in the
        e = int(np.argmax(c))
        for i in B:
            if A[i, e] > 0:
                Delta[i] = b[i] / A[i, e]
            else:
                Delta[e] = inf  # note that the program is bounded from above, since its objective function -\bar{x}_0, therefore it is bounded by 0.
        nonzero_indices = np.where(Delta > np.min(Delta))[0]
        l = int(nonzero_indices[np.argmin(Delta[nonzero_indices])])  # choosing the index for which Delta > 0 and minimum.
        N, B, A, b, c, nu = pivot(N, B, A, b, c, nu, l, e)  # perform a pivot

        x = np.zeros(n+1)  # initializing the solution array
        for i in range(n+1):
            if i in N:
                x[i] = b[i]

    if isclose(x[0], 0):   # x[n] corresponds to the value of additional variable added to the original LP
        if 0 in B:
            # perform a degenerate pivot to make the introduced variable non-basic
            nonzero_indices = np.where(A[0,N] != 0)
            e = nonzero_indices[0]  # picking one of the non-basic indices where A[0,e] != 0
            N, B, A, b, c, nu = pivot(N, B, A, b, c, nu, l=n+1, e=e)

        for i in B:
            if c_original[i] != 0:
                c_original += c_original[i]*(-A[i,:])
                nu += c_original[i]*b[i]
                c_original[i] = 0

        # remove the auxiliary variable from A, from the constraints and from the basic and non-basic sets
        A = A[1:,1:]
        b = b[1:]
        c = c_original[1:]
        N = {i - 1 for i in N if i != 0}
        B = {i - 1 for i in B}


        return N, B, A, b, c, nu
    else:
        return 'infeasible'



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
        e = int(np.argmax(c))
        for i in B:
            if A[i,e] > 0:
                Delta[i] = b[i] / A[i,e]
            else:
                Delta[e] = inf
        nonzero_indices = np.where(Delta > np.min(Delta))[0]
        l = int(nonzero_indices[np.argmin(Delta[nonzero_indices])])  # choosing the index for which Delta > 0 and minimum.
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
    nu = 0.0
    l = 5
    e = 0

    # Pivot test
    #N, B, A, b, c, nu = pivot(N=N, B=B, A=A, b=b, c=c, nu=nu, l=l, e=e)
    #print(f'N: {N}, B: {B}, A: {A}, c: {c}, nu: {nu}')

    # Simplex test
    solution, optimal_value = simplex(A, b, c)
    print("#------------- TESTS -------------")
    print(f'Optimal solution: {list(solution) == [8. , 4., 0. , 18., 0., 0.]}')
    print(f'Optimal value: {optimal_value == 28.0}')


    # non-feasible initial solution, testing initialize_simplex
    var_num = 4  # total number of variables
    temp = np.array([[2, -1], [1, -5]]).astype(float)
    A = np.zeros((var_num,var_num)).astype(float)
    A[2:,:2] = temp
    b = np.array([ 0, 0, 2, -4]).astype(float)
    c = np.array([2, -1, 0, 0]).astype(float)

    n = 4
    N, B, A, b, c, nu = initialize_simplex(A, b, c)
    B_exp = {1, 2}
    N_exp = {0, 3}
    temp = np.array([[-1/5, 0, 0, -1/5], [9/5, 0, 0, -1/5]])
    A_exp = np.zeros((n, n)).astype(float)
    A_exp[1:3,:] = temp
    b_exp = np.array([0, 4/5, 14/5, 0])
    c_exp = np.array([9/5, 0, 0, -1/5])
    nu_exp = -4/5

    print(f'Test initialize_simplex: {(B == B_exp) & (N == N_exp) & np.all(np.isclose(b,b_exp)) \
        & np.all(np.isclose(c,c_exp)) & np.all((isclose(nu,nu_exp))) & np.all(np.isclose(A,A_exp))}')


