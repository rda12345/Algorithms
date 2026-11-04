"""
Maximum subarray

Finds the sequence of array elements giving the maximum sum, by divide and conquer.
At each recursion level divides the array into two parts, finds the maximum sum in the left and right
subarrays and comnpares them to the maximum sum subarray which crosses the partition boundary.
"""
import numpy as np

def find_max_crossing_subarray(A: list[int], low: int, mid: int, high: int) -> tuple[int]:
    """
    Returns the maximum subarray in A[low:(high+1)] which elements cross mid array index
    Args:
        A (list): array of integers
    
    Parameters:
        max_left (int): index where the maximum subarray begins
        max_right (int): last index of the subarray (including)
        left_sum (int): maximum sum of the left side subarray (last index is mid)
        right_sum (int): maximum sum of the right side subarray (first index is mid + 1)
                  
    Returns:
        (tuple[int]): (max_left, max_right, left_sum + right_sum)
    """
    left_sum = -np.inf
    sum = 0
    for i in range(mid, low - 1, -1):
        sum += A[i]
        if sum > left_sum:
            left_sum = sum 
            max_left = i
    right_sum = -np.inf
    sum = 0
    for j in range(mid + 1, high + 1):
        sum += A[j]
        if sum > right_sum:
            right_sum = sum 
            max_right = j
    return (max_left, max_right, left_sum + right_sum)


def maximum_subarray(A: list, low: int, high: int) -> tuple[int]:
    """
    Returns the maximum subarray in A[low:(high+1)]

    Args:
        A (list): array of integers
        low (int): low index 
        high (int): high index
    
    Return: 
        (tuple): (low index, high index, total sum)
    """   
    # base case
    if low == high:
        return (low, high, A[low])
    # divide
    mid = (low + high)//2
    (left_low, left_high, left_sum) = maximum_subarray(A, low, mid)
    (right_low, right_high, right_sum) = maximum_subarray(A, mid + 1, high)
    (cross_low, cross_high, cross_sum) = find_max_crossing_subarray(A, low, mid, high)
    # concquer
    if left_sum >= right_sum and left_sum >= cross_sum:
        return (left_low, left_high, left_sum)
    elif right_sum >= left_sum and right_sum >= cross_sum:
            return (right_low, right_high, right_sum)
    else: 
        return (cross_low, cross_high, cross_sum)
            
if __name__ == '__main__':
    L = [2, -1, 3, 5, 10, -30, 5, 8, 9, -2]
    low, high, sum = maximum_subarray(L, 0, len(L) - 1)
    print(f'low index: {low}')
    print(f'high index: {high}')
    print(f'Maximum sum: {sum}')