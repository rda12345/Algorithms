#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bellman-Ford algorithm
The algorithm finds the shortest path in a wieghted graph.
In constract to Dijkstra, Bellman-Ford also works for graphs with negative
wieghts. 
The algorithm identifies negative cycles and bypasses them, returning the 
shortest path bypassing the negative cycles if one exists.


Complexity:
    The algorithm scales as O(V*E), where V and E are the total number of vertices and
    edges.
"""
from graph import *
from Dijkstra import SP


def BellmanFord(graph,source):
    
    distances = {}
    parents = {}
    inf = float('inf') 
    #initializing distances and parents
    for node in graph.nodes:
        distances[node] = inf
        parents[node] = None
    distances[source] = 0
    
    for _ in range(len(graph.nodes)):
        #loop over all the edges of the graph. graph.egdes[v] contains a list of
        #all the nodes neigh with an edge (node,neigh).
        converged = True
        for node in graph.nodes:
            for neigh in graph.edges[node]:        
                temp = relax(node,neigh,distances,parents,graph.weights,converged)
                if temp == False:
                    converged = False
            
        if converged:
            break
        
    for node in graph.nodes:
        for neigh in graph.edges[node]: 
            if distances[neigh] > distances[node] + graph.weights[(node,neigh)]:
                raise NameError('There is a negative in the graph')
    return parents, distances
                


    
def relax(u,v,d,p,w,c):
    """
    Relaxes the (directed) edge (u,v). If a change to the distances was made
    it returns False.
    
    Input:
        u: node, source
        v: node, destination
        d: dict, maps each node to the best (evaluated) distance from
                source to the node.
        p: dict, maps each  node to the parent node in the shortest path 
                from the source.
        w: dict, giving the weight of each edge.
        c: bool, keeps track weither the algorithm converged.
    """
    if d[v] > d[u] + w[(u,v)]:
        d[v] = d[u] + w[(u,v)]
        v.set_distance(d[v])
        p[v] = u
        return False
 
        
class Node(object):

    def __init__(self,name):
        '''
        name: str, the name of the node.
        '''
        self.name = name
        self.distance = None
        
    def get_name(self):
        ''' Returns the name of the node.'''
        return self.name
    
    def set_distance(self,dist):
        self.distance = dist
    
    def __str__(self):
        return self.name
    
def SP(graph,source,dest):
    """
    Returns a list containg the shortest path from the source to the destination
    
    Input:
        graph
        source: node, source node
        dest: node, destination node
    """
    
    shortest_path = [dest]
    parents, distances = BellmanFord(graph,source)
    node = dest 
    while node != source:
        shortest_path.append(parents[node])
        node = parents[node]
    return shortest_path
    
"""   
def print_situation(distances):
    print(f'-----------iteration----------')
    for key in distances.keys():
        print(f'node: {key} ; distance: {distances[key]}')

if __name__ == "__main__":
    
    ## Building graph
    g = Digraph()
    
    # Introducing nodes
    nodes = []
    for i in range(6):
        nodes.append(Node(str(i)))
    # Adding them to the graph
    for node in nodes:
        g.add_node(node)
        

    # Adding edges to the graph and building the adjacency matrix
    tuples = [(0,1),(2,1),(3,2),(4,3),(5,4),(0,5),(4,1),(1,3)]   
    weights = [10,1,-2,-1,1,8,-4,2]
    w ={}
    for i,tup in enumerate(tuples):
        g.add_edge(WeightedEdge(nodes[tup[0]],nodes[tup[1]],weights[i]))
        w[(nodes[tup[0]],nodes[tup[1]])] = weights[i]
    source = g.nodes[0]
    parents,distances = BellmanFord(g,source)
    print('Distances:')
    for node in nodes:        
        print(f'distances[{node}] = {distances[node]}')
    print('\n')
    print('Parents: ')
    for node in nodes:
        print(f'p[{node}] = {parents[node]}')
    print('\n')
    destination = g.nodes[3]
    shortest_path = SP(g,source,destination)
    print(f'Shortest path from: {source} --> {destination}: ')
    [print(f'{node} ', end=" ") for node in shortest_path[::-1]]
"""