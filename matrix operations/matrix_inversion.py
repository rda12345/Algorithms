#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Inversion - Based on chapter 28 of CLRS book


Includes three algorithms, used to implement an inversion of a non-singular matrix

LUP_solve: given a LUP decomposition of a square non-singular matrix, A,
         returns the solution x = A^{-1}*b
         complexity: O(n^2) where n=dim(A) (i.e, the number of rows, columns)
         
         
LU_decomposition: given a positive semi-definite matrix, A, returns the LU
                decomposition of the matrix, where L is a lower diagonal matrix with
                ones on the diagonal and U is an upper diagonal matrix.
                complexity: O(n^3)
            
 LU_decomposition: given a non-singluar matrix, A, returns the LUP
                 decomposition of the matrix, where L is a lower diagonal matrix with
                 ones on the diagonal and U is an upper diagonal matrix, and P is a diagonal matrix.
                complexity: O(n^3)
                
"""
import numpy as np

def LUP_solve(L, U, pi, b, n):
    """
    Solves the linear equation L*U*P*x = b.
    
    Parameters:
        L: np.ndarray (n,n),
        U: np.ndarray (n,n),
        P: np.ndarray (n,n),
        pi: list, indicates that P_{i,pi[i]}=1
        b: np.ndarray (n)
        n: int, dimension of L and U
    """
    # initiating x and y arrays
    x, y = np.zeros(n), np.zeros(n)
    
    # forward substitution
    for i in range(n):
        s = np.sum([L[i,j]*y[j] for j in range(i)])        
        y[i] = b[pi[i]] - s
    
    # backward substitution
    for i in reversed(range(3)):
        w = np.sum([U[i,j]*x[j] for j in range(i+1,n)])
        x[i] = (y[i] - w)/U[i,i]
     
    return x   
        
#TODO   
def LU_decomposition_out_of_place():
    pass


#TODO 
def LU_decomposition():
    pass

#TODO   
def LUP_decomposition():
    pass

        
if __name__ == "__main__":
    print("---------- Tests ----------")
    A = np.array([[1, 2, 0], [3, 4, 4], [5, 6, 3]])
    L = np.array([[1, 0, 0], [0.2, 1, 0], [0.6, 0.5, 1]])
    U = np.array([[5, 6, 3], [0, 0.8, -0.6], [0, 0, 2.5]])
    pi = [2,0,1]
    n = len(A)
    b = np.array([3, 7, 8])
    x = LUP_solve(L, U, pi, b, n)  
    print(f"x: {x}")
