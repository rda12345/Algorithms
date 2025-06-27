#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble sort
"""

# Function implementing bubble sort algorithm. The algorithm scales O(n^2)
def bubble_sort(L):
    '''
    Sorts the list by applying bubble sort
    
    L : list
        
    Returns a sorted list
    '''
    check = True
    while check:
    # Go over the list and each time a pair of elements and exchange them if nessecary.
        check = False
        for i in range(len(L)-1):
            # If the order is reversed, peform a swap between the elements.
            if L[i] > L[i+1]:
                L[i],L[i+1] = L[i+1],L[i]
                check = True
    return L     