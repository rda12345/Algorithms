#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth first search of the the shortest path betwwen two points on a graph
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
    #print('Path: ',[str(node) for node  in path])
    if start == end:
        return path
    elif shortest == None or len(path) < len(shortest):
        for node in graph.children_of(start):
            if node not in path: # Prevents entering loops
                new_path = DFS(graph,node,end,path,shortest)
                if new_path != None:
                        shortest = new_path
                        
    return shortest

"""
## Test
if __name__ == "__main__":

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
    tuples = [(0,1),(1,2),(2,3),(2,4),(3,4),(3,5),(0,2),(1,0),(3,1),(4,0),(0,5)]
    for tup in tuples:    
        g.add_edge(Edge(nodes[tup[0]],nodes[tup[1]]))

        
    shortest_path = DFS(g,nodes[0],nodes[5],[],shortest = None)
    
    print('DFS Shortest path: ',[str(x) for x in shortest_path],'\n ')
"""