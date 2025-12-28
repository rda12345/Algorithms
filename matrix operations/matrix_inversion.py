#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Inversion - Based on chapter 28 of CLRS book


Includes three algorithms, used to implement an inversion of a non-singular matrix

LUP_solve: given a LUP decomposition of a square non-singular matrix, A,
         returns the solution x = A^{-1}*b
         time complexity: O(n^2) where n=dim(A) (i.e, the number of rows, columns)
         
         
LU_decomposition: given a positive semi-definite matrix, A, returns the LU
                decomposition of the matrix, where L is a lower diagonal matrix with
                ones on the diagonal and U is an upper diagonal matrix.
                time complexity: O(n^3)
            
LUP_decomposition: given a non-singluar matrix, A, returns the LUP
                ecomposition of the matrix, where L is a lower diagonal matrix with
                ones on the diagonal and U is an upper diagonal matrix, and P is permutation matrix.
                time complexity: O(n^3)
                
"""
import numpy as np

def LUP_solve(L, U, pi, b, n):
    """
    Solves the linear equation L*U*P*x = b.
    
    Parameters:
        L: np.ndarray (n,n), lower diagonal matrix with ones on the diagonal
        U: np.ndarray (n,n), upper diagonal matrix
        P: np.ndarray (n,n), permutation matrix 
        pi: list, indicates that P_{i,pi[i]}=1, ecoding a permutation of the rows
        b: np.ndarray (n)
        n: int, dimension of L and U
    
    Returns:
        x, np.ndarray, the solution to the system of coupled linear equations
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
        
 
def LU_decomposition_out_of_place(M):
    """
    Performs an out of place LU decomposition
    
    Parameters:
        M: np.ndarray (n,n), non-singular matrix to be decomposed
        
    Return:
        L: np.ndarray (n,n), lower diagonal matrix with ones on the diagonal
        U:  np.ndarray (n,n), upper diagonal matrix
    """
    A = np.copy(M) 
    n = len(A)
    L, U = np.eye(n), np.zeros((n,n))
    
    for k in range(n):
        U[k,k] = A[k,k]
        for i in range(k+1,n):
            L[i,k] = A[i,k]/U[k,k]
            U[k,i] = A[k,i]
        # compute the Schur comnplement
        for i in range(k+1, n):
            for j in range(k+1, n):
                A[i,j] = A[i,j] - L[i,k]*U[k,j]
    return L, U

                
            


def LU_decomposition(M):
    """
    Performs an in place LU decomposition
    
    Parameters:
        M: np.ndarray (n,n), non-singular matrix to be decomposed
    
    Return:
        L: np.ndarray (n,n), lower diagonal matrix with ones on the diagonal
        U:  np.ndarray (n,n), upper diagonal matrix
    """
    A = np.copy(M)
    n = len(A)
    
    for k in range(n):
        for i in range(k+1,n):
            A[i,k] = A[i,k]/A[k,k]
        # compute the Schur comnplement
        for i in range(k+1, n):
            for j in range(k+1, n):
                A[i,j] = A[i,j] - A[i,k]*A[k,j]
    return A

    
#TODO   
def LUP_decomposition():
    """
    Performs an inplace LU decomposition
    
    Parameters:
        A: np.ndarray (n,n), non-singular matrix to be decomposed
    
    Return:
        L: np.ndarray (n,n), lower diagonal matrix with ones on the diagonal
        U:  np.ndarray (n,n), upper diagonal matrix
    """
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
    
    #A1 = np.array([[4,-5,6],[8,-6,7],[12,-7,12]])
    A1 = np.array([[2,3,1,5],[6,13,5,19],[2,19,10,23],[4,10,11,31]])
    L1, U1 = LU_decomposition_out_of_place(A1)
    #print(f"L = {L1}")
    #print(f"U = {U1}")
    print(f"L*U check: {np.max(L1@U1-A1) == 0.0}")
    
    modified_A= LU_decomposition(A1)
    print(f"modified A check: {np.max(modified_A - L1 + np.zeros(4) - U1 == 0.0)}")   
    
    