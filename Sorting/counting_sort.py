"""
Counting sort

The sorting algorithm assumes the sorted elements are integers in the range [0, k]. When k = O(n) the
run time of the algorithm is O(n).
The algorithm determines for each element x, the number of elements less than or
equal to x, and utilizes this information to position x.
It is a stable sorting algorithm, i.e., for equal elements, it preserves the original relative position of the elements.
The stability property is important when satellite data is carried around along with the integer keys. Moreover,
counting sort is utilized as a subroutine of radix sort. The stability property is important for radix sort
to sort properly.   
"""

def counting_sort(A: list, k) -> list:
    """
    Args:
        A (list): Input array
        k (int): The elements of the input array, A, are in the range [0, k]

    Returns:
        (list): sorted array
    """
    C = [0] * (k + 1)           # array holding the output
                                # provides temporary working storage
    B = [None] * len(A)
    for j in range(len(A)):
        C[A[j]] += 1
    # C[i] now contains the number of elements equal to i
    for i in range(1, k+1):
        C[i] += C[i-1]
    # C[i] now contains the number elements less than or equal to i
    for j in reversed(range(len(A))):
        B[C[A[j]]-1] = A[j]   
        C[A[j]] -= 1    # in case of duplicates, prevents equal elements from being stored in the same spot in B
    return B

"""
if __name__ == '__main__':
    import random
    L = [random.randint(1, 20) for i in range(20)]
    print(f'Original array: {L}')
    print(f'Sorted array: {counting_sort(L, 20)}')
"""