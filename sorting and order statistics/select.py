"""
Select algorithm for the selection problem (finds the i'th order statistic, i.e., the i'th
smallest element).

The algorithm recursively splits the array (similarly to randomized select), to 
find the desired order statistic. Unlike randomized select, select guarantees a good split
by choosing a good pivot when partitioning the array. It finds the pivot element recursively.

The algorithmic steps are:
1. Divide the n elements of the input array into ceil{n/5} groups of 5 elements each (or less for the last group)
2. Find the median of each of the groups by first insertion-sorting the elements of each group. Pick the median from the sorted list of group elements.
3. Use select recursively to find the median x of the ceil{n/5} medians (if there are an even number of medians, then by our convention, x is the lower median.)
4. Partition the input array around the median-of-medians x. Let k be one more than the number of elements on the low side of the partition,
 so that x is the kth smallest element and there are n-k elements on the high side of the partition.
5. If i=k, then return x. Otherwise, use select recursively to find the ith smallest element on the low side if i < k,
 or the (i - k)'th smallest element on the high side if i > k.

Time complexity: O(n) in the worst case.

Nevertheless, compared to randomized select algorithm it is mostly of theoretical interest, since
it is not as practical as randomized select.

"""
from insertion_sort import insertion_sort
import math

def partition_around(A: list, p: int, r: int, x: int) -> list:
    """
    Partitions the array A[p:(r+1)] into two (possibly empty) subarrays
    A[p:q] and A[q+1:(r+1)], such that the former includes the elements in A[p:(r+1)] which
    are smaller than x = A[q] and the latter includes the elements larger than x.
    
    Args: 
        A (list): array whose elements are reorganized
        p (int): lowest element's index
        r (int): highest element's index
        x (int): pivot
    
    Returns:
        (int): the position of the pivot

    Run time: O(n)
    """
    q = A.index(x, p, r + 1)    # locate the pivot's position in A[p:(r+1)]
    A[q], A[r] = A[r], A[q]
    i = p - 1   # highest index on the lowest side
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return (i + 1)

def select(A: list, p: int, r: int, i: int) -> int:
    """
    Returns the i'th order statistic of A[p : (r + 1)]

    Args:
        A (list): array of numbers
        p (int): index of first array element
        r (int): index of last array element
    
    Returns:
        i'th order statistic (i'th smallest element)
    """
    # reduce the number of elements in A[p:(r+1)] to be divisible by 5
    while ((r - p + 1) % 5) != 0:
        for j in range(p+1, r+1):
            if A[p] > A[j]:
                A[p], A[j] = A[j], A[p]
            # if we want the minimum of A[p:r], we are done
        if i == 1:
            return A[p]
        # otherwise, we want the (i-1)'th element of A[(p+1):(r+1)]
        p += 1
        i -= 1
    g = (r - p + 1) // 5     # number of 5 element groups
    # sort the remaining 5 element subgroups
    for j in range(p, p + g):
        idx = [j + k * g for k in range(5)]
        sorted_vals = insertion_sort([A[t] for t in idx])
        for a, v in zip(idx, sorted_vals):
            A[a] = v
    # all group medians now lie in the middle fifth of A[p:(r+1)]
    # find the pivot recursively as the median of the group medians
    x = select(A, p + 2 * g, p + 3 * g - 1, math.ceil(g/2))
    q = partition_around(A, p, r, x)    # partition around the pivot
    k = q - p + 1   # number of elements smaller and equal to x (including x)
    if i == k:
        return A[q]
    elif i < k:
        return select(A, p, q - 1, i)
    else:
        return select(A, q + 1, r, i - k)



if __name__ == '__main__':
    import random
    L = list(range(1, 10))
    i = 5
    print(f'List: {L}')
    print(f'i: {i}')
    print(f"i'th order statistic: {select(L, 0, len(L)-1, i)}")
