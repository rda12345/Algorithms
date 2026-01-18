#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three implementation of the fast fourier transform (FFT) algorithm.
1. Radix-2 recursive Cooley–Tukey fft algorithm.
2. Iterative fft (has a lower constant in the actual running time
                    not necessarily faster, depending on how the hardware
                    uses the cache).

A fft-shift function is also implemented, allowing easy plotting of the results.

The algorithms assume that the length of the input array is a power of 2.
Input and output are numpy arrays. 

The fft algorithm is a  divide and conquer method, where the recursion relation is based
on a separation to two subroutines on the even and odd indices, respectively,
and a reuse of the evaluated results. 

The fft (similarly DFT) is essentially a transformation between the coefficient
representation and the point-value representation of polynomials, where
the points of the polynomials are (n) complex roots of unity. The inverse transform
interpolates, mapping the n roots to the coefficients of teh polynomial.
Utilizing the fft algorithm allows multiplying polynomials in O(n*log(n)) (convolution theorem),
by performing an fft and then multiplying the f(x_i) values of each point, takes O(n),
and transforming back to the coefficient representation.

A parallel implementation of the fft algorithm utilizing specific circuit design
allows evaluating the fft with a circuit depth of O(log(n)).

The inverse transformation is obtained by taking a complex conjugate twiddle factor.


Complexity: O(n*log(n)), where n is the length of the coefficient vector.
"""
import numpy as np


def recursive_fft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of x 
    using the recursive Cooley–Tukey FFT algorithm. Input and output are numpy arrays.
    
    Parameters: 
        x: np.ndarray, input signal (length must be of power of 2) 
    
    Returns:
        X: np.ndarray: DFT of x     
    """
    n = len(x)  # array length
    
    # base case
    if len(x)==1: return x

    
    E = recursive_fft(x[::2])  # fft on all even elements of x
    O = recursive_fft(x[1::2]) # fft on all odd elements of x
    
    k = np.arange(0,n//2)     # fourier index vector
    W = np.exp(-1j*2*np.pi*k/n) # twiddle factor
    
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


def iterative_fft(x):
    A = bit_reverse_copy(x)  # A is a numpy array of length n in bit-reversed permutation order
    n = len(x)      # n is a power of 2
    for s in range(1,n.bit_length()):
        m = 2**s
        omega_m = np.exp(-1j*2*np.pi/m)     # twiddle factor
        for k in range(0,n,m):
            omega = 1
            for j in range(m//2):
                t = omega * A[k + j + (m//2)]
                u = A[k + j]
                A[k + j] = u + t
                A[k + j + (m//2)] = u - t
                omega = omega * omega_m

    return A

def bit_reverse_copy(x):
    """
    Bit reverses the element of an array.

    Args:
        x: np.ndarray, input array

    Returns:
        np.ndarray, bit reversed array of type complex
    """
    n = len(x)
    A = np.zeros(len(x), dtype=complex)
    m = (n-1).bit_length()  # number of bits in the greatest element of x
    for k in range(n):
        A[rev(k,m)] = x[k]
    return A

def rev(k, n):
    """
    Reverses the bits of an int.

    Args:
        k: int, the number to be bit reversed
        n: int, number of bits
    """
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | k & 1
        k >>= 1
    return rev



## Test
import matplotlib.pyplot as plt
if __name__ == "__main__":
    L = 1
    sig = L/10
    N = 2**6
    x = np.linspace(-L,L,N)
    k = np.arange(-N/2,N/2)*np.pi/L
    y = np.exp(-(x**2)/(2*sig**2))/(np.sqrt(2*np.pi*sig**2))



    fft = recursive_fft(y)
    shifted_fft = fftshift(fft) 
    plt.plot(k,np.abs(shifted_fft))
    plt.xlabel("k")
    plt.ylabel("|fft(x)|")
    plt.title("Absolute value of the DFT of a Gaussian")
    plt.grid(True)
    #plt.show()

    print('------------- TESTS -------------')

    ## Bit reversal test
    a = np.array(list(range(8)))
    a_new = bit_reverse_copy(a)
    a_res = np.array([0, 4, 2, 6, 1, 5, 3, 7])
    print(f"Bit reversal test: {np.array_equal(a_new, a_res)}")

    ## Iterative fft test
    iter_fft = iterative_fft(y)

    print(f"Iterative fft check: {np.max(np.imag(iter_fft-fft)) < 1e-14}")

    
    
