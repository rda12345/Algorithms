#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file includes sorting algorithms and evaluates thier running time.
Sorting algorithms
    1. Insertion sort, O(n^2)
    2. Bubble sort, O(n^2)
    3. Selection sort, O(n^2)
    4. Merge sort, O(n*ln(n))
    5. Heap sort, O(ln(n)) 
    6. AVL sort, O(ln(n))
    7. Quick sort
"""
import time
import pylab


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
        


def run_time(L,sort_method):
    '''
    Tests the running times of a sort algorithm, for different lengths of random floats
    
    L : list to be sorted
    sort_method : sorting method to be used
        
    Returns: running times and sorted list
    '''
    start_time = time.time()
    
    sorted_list = sort_method(L)
    # The running time in seconds
    run_time = time.time() - start_time
    return (run_time,sorted_list)


def plot_func(xVal,yVal,xLabel,yLabel,title):
    pylab.plot(xVal,yVal)
    pylab.title(title)
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    pylab.show()

# # Evaluate the run time of merge sort
# rt_merge_vec = []
# list_len = [10**3,10**5,10**3]
# index = 0
# for length in list_len:
#     L = numpy.random.random(length)
#     (rt_merge,list_merge) = run_time(L, merge_sort)
#     rt_merge_vec.append(rt_merge)
# pylab.figure(1)    
# plot_func(list_len,rt_merge_vec,'List Length','Running time','Merge Sort Running Times')
    
    

    
# rt_selection_vec = []
# list_len = [10**3,10**4,10**3]
# index = 0
# for length in list_len:
#     L = numpy.random.random(length)
#     (rt_selection,list_selection) = run_time(L, selection_sort)
#     rt_selection_vec.append(rt_selection)
# pylab.figure(2)    
# plot_func(list_len,rt_selection_vec,'List Length','Running time','Selection Sort Running Times')


            









    
    
    
    
    








