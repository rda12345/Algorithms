#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An implementation of the radix-2 recursive Cooley–Tukey FFT algorithm.
Assumes the length of the input array is a power of 2.
Input and output are numpy arrays. 

This is a divide and conquer algorithm, where the recursion relation is based
on a separation to two subroutines on the even and odd indices, respectively,
and a reuse of the evaluated results. 

Complexity: O(N*log(N))
"""
import numpy as np


def fft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of x 
    using the recursive Cooley–Tukey FFT algorithm. Input and output are numpy arrays.
    
    Parameters: 
        x: np.ndarray, input signal (length must be of power of 2) 
    
    Returns:
        X: np.ndarray: DFT of x     
    """
    N = len(x)  # array length
    
    # base case
    if len(x)==1: return x

    
    E = fft(x[::2])  # fft on all even elements of x
    O = fft(x[1::2]) # fft on all odd elements of x
    
    k = np.arange(0,N//2)     # fourier index vector
    W = np.exp(-1j*2*np.pi*k/N) # phases
    
    X_low = E + W*O     # DFT of elements k = [0,...,N/2-1] (using elementwise multipication of the phases and odd terms)
    X_high = E - W*O    # DFT of elements k = [N/2,...,N] 
    X = np.concatenate((X_low, X_high)) 
    return X
    
def fftshift(x):
    """
    Performs an FFT shift operation.

    Rearranges the output of an FFT so that the zero-frequency component 
    is centered in the array. 

    Originally, the frequency bins are ordered as: [0, 1, 2, ..., N/2 - 1, -N/2, ..., -1]
    After the shift, they are reordered to: [-N/2, ..., -1, 0, 1, ..., N/2 - 1]
    
    Note that the domain is periodic in N, therefore [-N/2, ..., -1, 0, 1, ..., N/2 - 1]
    is equivalent to [N/2, ..., -1, 0, 1, ..., N - 1].
    
    Parameters:
        x: np.ndarray, input array
        
    Returns:
        np.ndarray, shifted array
    """
    N = len(x)
    return np.concatenate((x[N//2:],x[:N//2]))

#TODO: complete
def iterative_fft(x):
    pass

#TODO: complete
def bit_reverse_copy(x, X):
    pass


## Test
import matplotlib.pyplot as plt
if __name__ == "__main__":
    L = 1
    sig = L/10
    N = 2**6
    x = np.linspace(-L,L,N)
    k = np.arange(-N/2,N/2)*np.pi/L
    y = np.exp(-(x**2)/(2*sig**2))/(np.sqrt(2*np.pi*sig**2))
    
    fft = fft(y)
    shifted_fft = fftshift(fft) 
    plt.plot(k,np.abs(shifted_fft))
    plt.xlabel("k")
    plt.ylabel("|fft(x)|")
    plt.title("Absolute value of the DFT of a Gaussian")
    plt.grid(True)
    plt.show()
    
    
    
    
    
