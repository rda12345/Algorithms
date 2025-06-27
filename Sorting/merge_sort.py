#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge sort
"""


# Function implementing insertion sort algorithm. The algorithm scales O(n*log(n))
def merge_sort(L):
    '''
    Sorts the list by applying bubble sort
    
    L : list
        
    Returns a sorted list
    '''
    # Check base case. Note that len(L) is O(1) in Python so there isn't any advantage in defining a seperate variable for the length's list.
    if len(L) == 1:  
        return L
    
    # Split the list to half
    q = len(L)//2
    L1 = L[:q]
    L2 = L[q:len(L)]
    
    # Sort both halfs of the list
    sorted_L1 = merge_sort(L1)
    sorted_L2 = merge_sort(L2)
    
    # Merge the sorted lists and return
    return merge(sorted_L1,sorted_L2)
    

def merge(L1,L2):
    '''
    L1 : list, sorted.
    L2 : list, sorted.

    Returns : merged list 
    '''
    merged_list = []
    # Using a two finger approach order the elements
    i = 0
    j = 0
    while i<len(L1) and j<len(L2):
        if L1[i]<=L2[j]:
            merged_list.append(L1[i])
            i += 1
        else:
            merged_list.append(L2[j])
            j += 1
    # Add the non-empty list to the end of the merged list  
    if i<len(L1):
        merged_list.extend(L1[i:])
    else:
        merged_list.extend(L2[j:])
    return merged_list