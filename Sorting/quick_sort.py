#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sort algorithm
"""
import random

def quick_sort(L,low,high):
    ''' Sorts the list L[low:high] a list employing Quick sort recursively.
    
        Input 
            L: list, to be sorted
            low: int, index the first element of the list
            high: int, index of the last element of the list
            
        Return
            sorted list
        '''
    # Base case
    current_length = high - low
    
    # Apply a variant of insertion sort when current length <= 3
    if current_length <= 3:
        for i in range(low,high):
            elem = L[i]
            j = i-1
            # For each element check if the elements to before him are larger than him.
            # If so shift the elements right.
            while j>=0 and L[j]>elem:    
                L[j+1] = L[j]
                j -= 1
            # When L[j]!>elem or got to the end of the list insert the element at the appropriate position.
            L[j+1] = elem
            
        return
    
    # Initialize: Find a pivot by taking the midian the first, last and middle element.
    pivot= find_pivot(L,low,high)

    # Repeat until left.index = right_index + 1
    i = low
    j = high - 2
    while i < j+1:
        # Find the left holder > pivot
        while L[i] <= pivot and i < high - 1:
            i += 1
        #l L[i] is the left handle
        # Find the right holder < pivot
        while L[j] >= pivot and j > -1:
            j -= 1 
        #  L[j] right_handle
        # Swap left and right holders
        if i < j:
            L[i], L[j] = L[j], L[i]
    # Replace the pivot and the left holder

    L[i], L[high - 1] = L[high - 1], L[i]
    
    # Recurse on each one of the shorter lists
    quick_sort(L,low,j+1)
    quick_sort(L, i+1, high)



def find_pivot(L,low,high):
    ''' Sorts three elements of L and returns a tuple containing list and the pivot
        Input 
            L: list
        Retrun
            A tuple containing a modified list and the pivot
            In the modified list L[0],L[len(L)//2],L[len(L)] are first sorted and then 
            the midian and the last element of the sorted list are swapped;
            The pivot is midian of the these three elements.
    '''
    mid = (high-low)//2+low
    end = high - 1
    
    for i in (low,mid):
        if L[end] < L[i]:
            L[end], L[i] = L[i], L[end]
    if L[mid] < L[low]:
        L[mid], L[low] = L[low], L[mid]
    L[mid], L[end] = L[end], L[mid]

    return L[end]




## Test 
#L = [2,6,5,3,8,7,1,0,4]
L = [random.randint(1, 20) for i in range(20)]
print(L)
quick_sort(L, 0, len(L))
print(L)

