#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary search, O(ln(n))
"""

def binary_search(L,val,start,end):
    ''' 
    Searches for the item val in the sorted list L.
    
    L: list, sorted from smallest to the largest element.
    start: first index of the list (set to 0)
    end: last index of the list (set to len(L)-1)
    val: object 
    
    return: tuple (bool, int), True if L[int] == val
    
    '''
    mid = start + (end-start)//2
    print('end: ',end)
    print('start: ', start)
    print('mid: ', mid)
    if L[mid] == val:
        return (True,mid)
    elif start == end:
        return False
    elif L[mid] < val:
        return binary_search(L, val,mid + 1,end)
    else:
        return binary_search(L,val,start,mid -1)
        
## Test    
L = [1,2,3,5,7,8,9]
print('result: ', binary_search(L,2,0,len(L)-1))