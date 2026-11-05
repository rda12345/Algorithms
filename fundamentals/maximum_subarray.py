"""
Maximum subarray

We present two algorithms one is a divide and conquer algorithm, and the other employs dynamical programming.
The algorithm find the subarray which sum is maximum.

In the divide and conquer algorithm at each recursion level divides the array into two parts,
finds the maximum sum in the left and right subarrays and comnpares them to the maximum sum subarray
which crosses the partition boundary.

In the dynamical programming solution we keep track on two quantities:
    - The maximum subarray of A[1:j]
    - The maximum subarray ending in the j'th element
To evaluate the maximum subarray of A[1:(j+1)], we note it is either the maximum subarray of A[1:j]
or the maximum subarray of A[i:(j+2)]. To update the maximum subarray ending in the (j+1)'th element, 
notice that it either includes the elements A[i:(j+2)] for i<=j or only the (j+1)'th element. 


Time complexity: The divide an conquer algorithm as a run time of O(n*log(n)).
    Each call divides the problem to two and requires O(n) work to find max
    crossing subarray, therefore the recursion relation is T(n) = 2*T(n/2) + O(n), which gives the
    n*log(n) run time.
    The dynamical programming algorithm has a run time of O(n), the for loop runs through the elements
    and each update rule costs O(1) time. 
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



def maximum_subarray_dp(A: list, low: int, high: int) -> tuple[int]:
    """
    Evaluates the maximum subarray by employing dynamical programming.

    
    """
    if len(A) == 0:
        raise ValueError('Array is empty')
    
    current_max = [0, 0, A[0]]  # the tuple corresponds to (first subarray index, last subarray index, sum)
                                # current_max will hold the maximum subarray in A[1:(j+1)]
    if len(A) == 1:
        return current_max
    crossing_max = current_max.copy()
    for j in range(1, len(A)):
        # evaluate maximum subarray ending at j

        if crossing_max[2] >= 0:    # add A[j] to the crossing array (crossing_max[2] + A[j] >= A[j])
            crossing_max[1] += 1
            crossing_max[2] += A[j]
        else:   # start a new subarray consisting of A[j]
            crossing_max = [j, j, A[j]]
        # if the maximum subarray of A[1:(j+1)] consistes of a crossing array A[i:(j+1)],
        # replace current maximum subarray
        if crossing_max[2] >= current_max[2]:
            current_max = crossing_max.copy()

    return current_max
            
    
            

            
if __name__ == '__main__':
    L = [2, -1, 3, 5, 10, -30, 5, 8, 9, -2]

    print(f'List: {L}')

    print('Divide and conquer algorithm')
    low, high, sum = maximum_subarray(L, 0, len(L) - 1)
    print(f'low index: {low}')
    print(f'high index: {high}')
    print(f'Maximum sum: {sum}\n')

    print('Dynamical programming algorithm')

    low, high, sum = maximum_subarray_dp(L, 0, len(L) - 1)
    print(f'low index: {low}')
    print(f'high index: {high}')
    print(f'Maximum sum: {sum}\n')