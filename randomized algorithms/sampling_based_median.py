"""
Sampling-Based Median Finding Algorithm

Given an unsorted array of n distinct numbers, we want to find the median (the middle element in the sorted order).
If the algorithm doesn't output the 'FAIL' string, it is guaranteed to output the correct median.
The considered version assumes all numbers are distinct and n is odd.       

Complexity: O(n) (2n + o(n))) expected time. The expected runtime can be improved to 3n/2 + o(n) by using a more careful sampling strategy.
"""
import numpy as np

def sampling_based_median(arr: np.ndarray) -> int | str:
    """
    Returns the median of the input array, or 'FAIL' if the algorithm fails.

    Args:
        arr: list of distinct integers

    Returns:
        int: the median of the input array, if the algorithm succeeds or
        str: 'FAIL' if the algorithm fails
    """
    n = len(arr)
    if n % 2 == 0:
        raise ValueError("Input array must have an odd number of distinct elements.")
    
    sample_size = int(n**(3/4))
    sample = np.random.choice(arr, size=sample_size, replace=False)
    sample  = np.sort(sample)  # sort the sample to find the median of the sample efficiently
    
    a = sample[int(sample_size // 2 - np.sqrt(n))]  
    b = sample[int(sample_size // 2 + np.sqrt(n))]  
    # compare each element in the array to a and b, and count how many elements are less than a, between a and b, and greater than b
    T = []
    count_less, count_greater = 0, 0
    for x in arr:
        if x < a:
            count_less += 1
        elif x > b:
            count_greater += 1
        else:
            T.append(x)
    T = np.array(T)
    
    if count_less < n//2 and count_greater < n//2 and len(T) <= 4*sample_size:
        T = np.sort(T)
        return T[(n+1)//2 - count_less - 1]
    else: 
        return 'FAIL'
    

if __name__ == "__main__":
    n = 501
    arr = np.random.choice(range(1, 1000), size=n, replace=False)
    median = sampling_based_median(arr)
    print(f"Median: {median}")
    print(f"Actual median: {int(np.median(arr))}")