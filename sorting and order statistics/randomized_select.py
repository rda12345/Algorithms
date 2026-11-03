"""
Randomized select

A randomized algorithm to find the i'th order statistic.

The i'th order statistic is the i'th smallest element, e.g., the first order statistic is the minimum.
The selection problem is defined as follows, given a set A of distinct numbers and an integer
1 \leq i \leq n, return the element x which is larger than i-1 other elements of A.

Time complexity: worst case of O(n^2), where n = len(A). Each time the pivot can be the minimum or maximum element,
randomized partition then takes O(n) time, giving the recursion relation T(n) = T(n-1) + O(n).
The expected runtime is O(n), since the probability of choosing a "good" pivot element in randomized partition subroutine
is >1/2. 
The run time of randomized partition is O(r-p).
"""
import random

def randomized_partition(A: list, p: int, r: int) -> list:
    """
    Partitions the array A[p:(r+1)] into two (possibly empty) subarrays
    A[p:q] and A[q+1:(r+1)], such that the former includes the elements in A[p:(r+1)] which
    are smaller than A[q] and the later induces the elements larger than A[q].
    The element A[q] (not included in either of the subarrays) is the denoted as the pivot. 
    
    Args: 
        A (list): arrray which elements are reorganized
        p (int): lowest element's index
        r (int): highest element's index
    
    Returns:
        (int): the position of the pivot

    Run time: O(n)
    """
    l = random.randint(0, r)
    A[l], A[r] = A[r], A[l]     
    x = A[r]    # the pivot
    i = p - 1   # highest index on the lowest side
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return (i + 1)

    
def randomized_select(A: list, p: int, r: int, i: int):
    """
    Returns the i'th order statistic of A[p : (r + 1)]

    Args:
        A (list): array of numbers
        p (int): index of first array element
        r (int): index of last array element
    
    Returns:
        i'th order statistic (i'th smallest element)
    """
    if p == r:
        return A[p]
    q = randomized_partition(A, p, r)   # the pivot
    k = q - p + 1   # the number of elements less or equal to the pivot, including the pivot
    if i == k:  # check wether the pivot, A[q], is the i'th order statistic
        return A[q]
    elif i < k: # more elements on the left partition than i
        return randomized_select(A, p, q-1, i)  
    else:   # i'th order statistic is one of the elements larger than the pivot element
        return randomized_select(A, q+1, r, i - k)


if __name__ == '__main__':
    i = 4
    L = [2, 8, 7, 1, 3, 5, 6, 4]
    print(f'Random partition test:')
    print(f'List: {L}')
    print(f'Pivot position: {randomized_partition(L, 0, len(L)-1)}')
    print(f'List after partition: {L}\n')

    print(f"i'th order static test:")
    print(f'i: {i}')
    print(f"i'th order statistic: {randomized_select(L, 0, len(L)-1, i)}")