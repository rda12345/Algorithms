#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An implementation of the Strassen algorithm for matrix multipication and a
comparison to the standard loop algorithm.
"""
import numpy as np
import time

def strassen(A,B):
    '''
    Evaluates C+A*B bu the Strassen algorithm for matrix multipication. 
    
    Input: 
        A,B,C: array
        n: int, size of the arrays, must by a power of 2
    Return: 
        C+A*B
    '''
    n = len(A)
    if n <= 2:
        return np.dot(A,B)

    mid = n//2
    A11 = A[0:mid,0:mid]
    A12 = A[0:mid,mid:n]
    A21 = A[mid:n,0:mid]
    A22 = A[mid:n,mid:n]
    
    B11 = B[0:mid,0:mid]
    B12 = B[0:mid,mid:n]
    B21 = B[mid:n,0:mid]
    B22 = B[mid:n,mid:n]
    
    # C11 = C[0:mid,0:mid]
    # C12 = C[0:mid,mid:n]
    # C21 = C[mid:n,0:mid]
    # C22 = C[mid:n,mid:n]
    
    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10 = B11 + B12
    
    
    P1 = strassen(A11,S1)
    P2 = strassen(S2,B22)
    P3 = strassen(S3,B11)
    P4 = strassen(A22,S4)
    P5 = strassen(S5,S6)
    P6 = strassen(S7,S8)
    P7 = strassen(S9,S10)
    
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7
    
    return np.vstack((np.hstack((C11,C12)),np.hstack((C21,C22))))


def matrix_mult(A,B):
    n = len(A)
    C = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j]
    return C


## Test

start_time = time.time()
m = 9
N = 2**m
A = np.random.randint(10, size=(N,N))
B = np.random.randint(10, size=(N,N))

start_time = time.time()
C = strassen(A,B)
# The running time in seconds
run_time_strassen = time.time() - start_time
print('Strassen algo running time: ',run_time_strassen)

start_time = time.time()
C = matrix_mult(A,B)
# The running time in seconds
run_time_standard = time.time() - start_time
print('Standard algo running time: ',run_time_standard)

#print(sum(sum(abs(A@B-strassen(A,B)))))

    
    