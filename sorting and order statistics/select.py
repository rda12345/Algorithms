"""
Select algorithm for the selection problem (finds the i'th order statistic, i.e., the i'th
smallest element).
"""
# TODO: complete both the explaination and the code, verify the code
def select(A: list, p: int, r: int, i: int):
    """
    Returns the i'th order statistics of A[p : (r + 1)]

    Args:
        A (list): array of numbers
        p (int): index of first array element
        r (int): index of last array element
    
    Returns:
        i'th order statistics (i'th smallest element)
    """
    pass



if __name__ = '__main__':
    import random
    L = [random.randint(1, 20) for i in range(20)]
    i = 9
    print(f'List: {L}')
    print(f'i: {i}')
    print(f"i'th order statistics: {randomized_select(L, 0, len(L)-1, i)}")
