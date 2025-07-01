#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floyd-Warshall algorithm

Returns the shortest path between all pairs of vertices, where negative weights
are allowed.
Assumes there aren't any negative cycles.

Complexity:  O(V^3)
"""
from graph import *
import numpy as np

def FloydWarshall(graph):
    """
        Returns the shortest path between all pairs of vertices, where negative weights
        are allowed.
        
        Parameters: 
            d: array, |V| by |V| array containing the shortest distance between
                        corresponding vectors.
                        d[i,j] = shortest path from node i to node j.
    """
    # Initializization
    V = len(graph.nodes)        # number of nodes in the graph
   
    inf = 2**50             # a arbitrary large number which should be much greater
                            # than than any of the weights (approximation for infinity)
    distances = np.ones((V,V))*inf
    for node in graph.nodes:
        node_index = graph.nodes.index(node)
        distances[node_index, node_index] = 0
    #fill in the weights of each edge
    for edge in graph.weights.keys():
        src, dest = edge
        src_index = graph.nodes.index(src)
        dest_index = graph.nodes.index(dest)
        distances[src_index,dest_index] = graph.weights[edge]
    
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if distances[i,j] > distances[i,k] + distances[k,j]:
                    distances[i,j] = distances[i,k] + distances[k,j]
    return distances
                    
                    
"""                 
if __name__ == "__main__":
    
    ### Building graph
    g = Digraph()
    
    # Introducing nodes
    nodes = []
    for i in range(1,5):
        nodes.append(Node(str(i)))
    # Adding them to the graph
    for node in nodes:
        g.add_node(node)        

    # Adding edges to the graph and building the adjacency matrix
    tuples = [(1,3),(3,4),(4,2),(2,1),(2,3)]   
    weights = [-2,2,-1,4,3]
    
    for i,tup in enumerate(tuples):
        g.add_edge(WeightedEdge(nodes[tup[0]-1],nodes[tup[1]-1],weights[i]))
        
    distances = FloydWarshall(g)
    print('Distances:')
    print(distances)
    print('\n')
    print('Order of nodes:')
    [print(node, end = " ") for node in g.nodes]
"""   
    
                    
        