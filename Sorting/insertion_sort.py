#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insertion sort
"""

# Function implementing insertion sort algorithm. The algorithm scales O(n^2)
def insertion_sort(L):
    '''
    Sorts the list by applying insertion sort
    
    L : list
    
    Returns a sorted list
    '''
    # Run over the elements of the list
    for i in range(1,len(L)):
        elem = L[i]
        j = i-1
        # For each element check if the elements to before him are larger than him.
        # If so shift the elements right.
        while j>=0 and L[j]>elem:    
            L[j+1] = L[j]
            j -= 1
        # When L[j]!>elem or got to the end of the list insert the element at the appropriate position.
        L[j+1] = elem
    return L   