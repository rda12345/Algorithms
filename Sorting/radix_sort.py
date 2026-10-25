"""
Radix sort

The algorithm used in card sorting machines (which where used in the past).
It sorts by first sorting the least significant bit, then sorts the second least
significant bit and so on, until the all the integers are sorted. 
For the algorithm to sort properly, the sorting of each digits should be stable (for equal digits the order is maintained). 
It is implemented with counting sort algorithm (O(n + k) run time for stable integer sorting in the range [0, k]).

Run time: O(d(n + k)) to sort n d digit numbers, where each digit is in the range [0, k].
"""

def counting_sort_indicies(A: list[int], k:int ) -> list[int]:
    """
    Args:
        A (list): Input array
        k (int): The elements of the input array, A, are in the range [0, k]

    Returns:
        (list): indicies of sorted array, the j'th element contains the position of A[j] in the sorted array
    """
    C = [0] * (k + 1)           # array holding the output
                                # provides temporary working storage
    B = [None] * len(A)
    indicies = B.copy()
    for j in range(len(A)):
        C[A[j]] += 1
    # C[i] now contains the number of elements equal to i
    for i in range(1, k+1):
        C[i] += C[i-1]
    # C[i] now contains the number elements less than or equal to i
    for j in reversed(range(len(A))):
        C[A[j]] -= 1    # in case of duplicates, prevents equal elements from being stored in the same spot in B
        B[C[A[j]]] = A[j]   
        indicies[j] = C[A[j]] 
    return indicies

def radix_sort(A: list[int]) -> list[int]:
    """
    Args:
        A (list[int]): the list to be sorted
    Returns:
        (list[int]): sorted list
    """
    # find the number of digits
    d = 0
    for num in A:
        while num // (10**d) > 0:
            d += 1

    
    for exp in range(d):
        digits = [(num // (10**exp)) % 10 for num in A]
        indices = counting_sort_indicies(digits, 9)
        temp = [None] * len(A)
        for j in range(len(A)):
            temp[indices[j]] = A[j]
        A = temp
    return A

        
"""
if __name__ == '__main__':
    import random
    L = [random.randint(0, 999) for i in range(10)]
    print(f'Original array: {L}')
    print(f'Sorted array: {radix_sort(L)}')
"""