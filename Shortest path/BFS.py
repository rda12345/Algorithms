#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth first search of the the shortest path betwwen two points on a graph
"""
from graph import Node,Edge,Graph
from queue import Queue

def BFS(graph,start,end,path,shortest = None):
    ''' The function conductions a breadth first search algorithm to find the shortest 
        path between start and end nodes. The algorithm uses a the python ...
        to implement the queue abstract data structure
        
        graph: Graph
        start: Node, the initial node.
        end: Node, the final node.
        path: list, the list contains all the nodes along the path.
        
        return: shortest path if exists.
    '''
    path = [start]
    path_queue = Queue(maxsize=0) #A queue data structure with no limited size
    path_queue.put(path) # get insert an item in the queue
    while path_queue.qsize() > 0: 
        temp_path = path_queue.get()    # get returns and removes the first item in the queue
        for node in graph.children_of(temp_path[-1]):
            # Add the nodes to the queue
            if node not in temp_path:  # To prevent loops
                new_path = temp_path + [node]
                #print('Path: ', [str(x) for x in new_path])
                path_queue.put(new_path)
                if node == end:
                    if shortest == None or len(new_path) < len(shortest):
                        shortest = new_path        
    return shortest
            
## Test

if __name__ == "__main__" :
    
    ## Building graph
    g = Graph()
    # Introducing nodes
    nodes = []
    for i in range(6):
        nodes.append(Node(str(i)))
    # Adding them to the graph
    for node in nodes:
        g.add_node(node)
    



    # Adding edges to the graph
    tuples = [(0,1),(1,2),(2,3),(2,4),(3,4),(3,5),(0,2),(1,0),(3,1),(4,0),(0,5)]
    for tup in tuples:    
        g.add_edge(Edge(nodes[tup[0]],nodes[tup[1]]))
        

    shortest_path = BFS(g,nodes[0],nodes[5],[],shortest = None)
    print('BFS Shortest path: ',[str(x) for x in shortest_path])