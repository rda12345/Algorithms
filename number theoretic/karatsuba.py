"""
Karatsuba

Fast divide and conquer algorithm to multiply large numbers.
Separates a n-digit number to a high and low parts of two numbers:
x = x_high * (2**n/2) + x_low, and utilizes three recursive
multiplications instead of four for numbers split in half.


Complexity: O(n^{lg(3)})
"""
from scipy.special.cython_special import y1, y0


def karatsuba(x, y):
    """
    Multiplies two integer numbers efficiently
    """
    if x < 2 or y < 2:
        return x * y

    # Evaluate the number of digits in the high and low
    m = max(x.bit_length(), y.bit_length()) // 2

    x_high = x // (2**m)
    x_low = x % (2**m)
    y_high = y // (2 ** m)
    y_low = y % (2 ** m)

    z_low = karatsuba(x_low, y_low)
    z_mid = karatsuba(x_low + x_high, y_low + y_high)
    z_high = karatsuba(x_high, y_high)

    return z_high * 2**(2*m) + (z_mid - z_high - z_low) * (2**m) + z_low

if __name__ == '__main__':
    print(f'10*20: {karatsuba(10, 20) == 200}')
    print(f'3*5: {karatsuba(3, 5) == 15}')