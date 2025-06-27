#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selection sort
"""

# Function implementing insertion sort algorithm. The algorithm scales O(n^2)
def selection_sort(L):
    '''
    Sorts the list by applying selection sort
    
    L : list
        
    Returns a sorted list
    '''
    for place in range(len(L)-1,0,-1):
        max_elem = L[0]
        index = 0
        for i in range(place +1):
            if L[i] > max_elem:
                max_elem = L[i]
                index = i
        L[index],L[place] = L[place],L[index]   
    return L   