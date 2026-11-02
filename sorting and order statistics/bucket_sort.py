"""
Bucket_sort

A sorting algorithm, with a an average case runtime of O(n).
It assumes the input is randomly distributed in the interval [0, 1).
The algorithm divides the interval into n equal size subintervals, or buckets. It then sorts
the numbers in each bucket.

Time complexity: average case of O(n).
Space complexity: An additional array
"""
# TODO: complete explainiation algorithm and testing
from insertion_sort import insertion_sort

def bucket_sort(A: list[float]) -> list[float]:
    """
    Args:
        A (list): containing elements in the range [0, 1)

    Returns:
        (list): sorted array
    """
    n = len(A)
    B = [[] for i in range(n)]
    for num in A:
        B[floor(n * num)].append(num)
    # sort the list
    for i in range(n):
        B[i] = insertion_sort(B[i])
    return [B[i,j] for j in range(len(B[i])) for i in range(n)]



if __name__ == '__main__':
    import random
    L = [random.rand() for i in range(20)]
    print(f'Original array: {L}')
    print(f'Sorted array: {bucket_sort(L)}')