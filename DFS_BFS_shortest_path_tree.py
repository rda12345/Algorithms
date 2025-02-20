#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file includes
    1. Breadth first search of the the shortest path betwwen two points on a graph
    2. Depth first search of the the shortest path betwwen two points on a graph
"""
from graph import Node,Edge,Graph


def DFS(graph,start,end,path,shortest = None):
    ''' The function conductions a depth first search algorithm to find the shortest 
        path between start and end nodes. The algorithm includes a recursive 
        procedure which bypasses loops.
        
        graph: Graph
        start: Node, the initial node.
        end: Node, the final node.
        path: list, the list contains all the nodes along the path.
        
        return: shortest path if exists.
    '''
    path = path + [start]
    print('Path: ',[str(node) for node  in path])
    if start == end:
        return path
    elif shortest == None or len(path) < len(shortest):
        for node in g.children_of(start):
            if node not in path: # Prevents entering loops
                new_path = DFS(graph,node,end,path,shortest)
                if new_path != None:
                        shortest = new_path
                        
    return shortest


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
    path_queue = [path]
    while len(path_queue) > 0: 
        temp_path = path_queue[0]
        for node in g.children_of(temp_path[-1]):
            # Add the nodes to the queue
            if node not in temp_path:  # To prevent loops
                new_path = temp_path + [node]
                print('Path: ', [str(x) for x in new_path])
                path_queue.append(new_path)
                if node == end:
                    if shortest == None or len(new_path) < len(shortest):
                        shortest = new_path 
        # Remove the first path
        path_queue.pop(0)        
    return shortest
            
## Test

### Building graph

g = Graph()
# Introducing nodes
nodes = []
for i in range(6):
    nodes.append(Node(str(i)))
# Adding them to the graph
for node in nodes:
    g.add_node(node)
    


# Adding edges to the graph
g.add_edge(Edge(nodes[0],nodes[1]))
g.add_edge(Edge(nodes[1],nodes[2]))
g.add_edge(Edge(nodes[2],nodes[3]))
g.add_edge(Edge(nodes[2],nodes[4]))
g.add_edge(Edge(nodes[3],nodes[4]))
g.add_edge(Edge(nodes[3],nodes[5]))
g.add_edge(Edge(nodes[0],nodes[2]))
g.add_edge(Edge(nodes[1],nodes[0]))
g.add_edge(Edge(nodes[3],nodes[1]))
g.add_edge(Edge(nodes[4],nodes[0]))
g.add_edge(Edge(nodes[0],nodes[5]))
        

## Depth first search
shortest_path = DFS(g,nodes[0],nodes[5],[],shortest = None)
print('DFS Shortest path: ',[str(x) for x in shortest_path],'\n ')
## Breadth first search
shortest_path = BFS(g,nodes[0],nodes[5],[],shortest = None)
print('BFS Shortest path: ',[str(x) for x in shortest_path])
       
