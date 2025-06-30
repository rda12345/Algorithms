#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dijkstra algorithm. 
The algorithm finds the shortest paths from the source to all vertices of
a wieghted graph with positive weights.
It is a greedy algorithm.

Complexity: 
    The algorithm involves:
        - O(V) insertion into a queue
        - O(V) extracting the minimum element of the queue.
        - O(E) relaxations (each edge is at most relaxed a single time).
        
    
    Three ways of representing the queue (given in an order of improving complexity)
    - In an array reprentation of the queue extracting the minimum is O(V)
         As a result, the algorithm's running time is O(V^2 + E).
    - In a binary heap representation of the queue, insertions and extraction
        of the minimum element are O(log(V)).
        As a result, the algorithm's running time is O(V*log(V+ E*log(V)).
    - Using a Fibonacci heap, insertions are log(V) but extraction of the minimum
        is O(1).
        Aa result, the algorithm's running time is O(V*log(V)+E).
        This is the best known running time of Dijkstra.
        
"""
from graph import *

def dijkstra(graph,source):
    """
    Returns the shortes path from the source node
    to all connected vertices of a graph with non-negative wieghts.
    
    Input:
        graph: graph, the graph which the search is conducted in.
        source: node, the source node
        
    Parameters: 
        weights: dict, maps tuples of nodes corresponding to edges to the edge weight
    
    Returns:
        parents: dict, provides the parent of each node in the shortest path
                        from the source.
        distances: dict, provides the shortest path distance to all nodes.
    """
    distances = {}
    parents = {}
    inf = 2**50  
    #initializing distances and parents
    for node in graph.nodes:
        distances[node] = inf
        parents[node] = None
    distances[source] = 0
    
    #creating a weights dictionary
    #for edge in graph.edges:
        #print(edge)
        #print(type(edge))
        #weights[(edge.get_source(),edge.get_destination())] = edge.get_weight()
    
    frontier = Heap()
    frontier.insert(source)

    while frontier.n > 0:
        node = frontier.extract_min()
        for neigh in graph.edges[node]:
            relax(node,neigh,distances,parents,g.weights)
            if neigh not in frontier.L:
                frontier.insert(neigh)
    return parents, distances                
                
   

def relax(u,v,d,p,w):
    """
    Relaxes the (directed) edge (u,v).
    
    Input:
        u: node, source
        v: node, destination
        d: dict, maps each node to the best (evaluated) distance from
                source to the node.
        p: dict, maps each  node to the parent node in the shortest path 
                from the source.
        w: dict, giving the weight of each edge.
    """
    if d[v] > d[u] + w[(u,v)]:
        d[v] = d[u] + w[(u,v)]
        v.set_distance(d[v])
        p[v] = u
        
def SP(graph,source,dest):
    """
    Returns a list containg the shortest path from the source to the destination
    
    Input:
        graph
        source: node, source node
        dest: node, destination node
    """
    
    shortest_path = [dest]
    parents, distances = dijkstra(graph,source)
    node = dest 
    while node != source:
        shortest_path.append(parents[node])
        node = parents[node]
    return shortest_path
        
    
    
    
 
        
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
    
    def __str__(self):
        return self.name 
    
    def set_distance(self,dist):
        self.distance = dist
        
    
    def get_distance(self):
        return self.distance        
    
class Heap(object):
    """Min-heap data structure, ordered by the distance of the node."""
    
    def __init__(self,L = []):
        '''L is a python list'''
        self.L = L
        self.n = len(L)
    
    def __str__(self):
        return str(self.L)
        
    def min_heapify(self,n,i):
        """Heapifies the i'th node according to the distance of the node and its children."""
        left = 2*i + 1     # left child index
        right = 2*i + 2     # right  child indes
        smallest = i
        
        if left < self.n and self.L[left].get_distance() < self.L[smallest].get_distance():
            smallest = left
            
        if right < self.n and self.L[right].get_distance() < self.L[smallest].get_distance():
            smallest = right
        
        if smallest != i:
            self.L[smallest], self.L[i] = self.L[i], self.L[smallest]
            self.min_heapify(n,smallest)

    def min_heapify_list(self):
        '''Min heapifies the entire list'''        
        for i in range(self.n//2-1,-1,-1):
            self.min_heapify(self.n,i)
            
    def delete_node(self,index):
        '''Deletes the element L[index]. Algorithm swaps L[index] with the last
            leaf and heapifies the list. 
        '''
        if index < self.n:
            self.L[index], self.L[self.n-1] = self.L[self.n-1], self.L[index]
            self.L.pop()
            self.n -= 1
            self.min_heapify_list()
        else:
            print('Index out of range')
            
    def peek(self):
        '''Returns the root node (the maximum for a max heap and the minimum
            for a min-heap).
        '''
        if len(self.L) > 0:
            return self.L[0] 
        return None            
    
    def extract_min(self):
        '''Extracts the maximum item, which is located at the root of the tree,
            assuming the list is max-heapified.
        '''
        temp = self.L[0]
        self.delete_node(0)
        return temp
    
    def insert(self,item):
        '''Inserts the item in the last leaf and heapifies the tree.'''
        self.L.append(item)
        self.n += 1
        self.min_heapify_list()
        
    def check_ri(self,ind):
        """Checks the representation invariant"""
        left = 2*ind + 1
        right = 2*ind + 2
        
        if left < self.n:  
            if self.L[ind].get_distance() < self.L[left].get_distance():
                return self.check_ri(left)
            else:
                raise NameError(f'Left child, {l[left]}, violates the rep. inv.')
        elif right < self.n:
            if self.l[ind] > self.l[right]:
                return self.check_ri(self,right)    
            else:
                raise NameError('Right child, {l[right]}, violates the rep. inv.')
        return True
    
 
if __name__ == "__main__":
    
    ### Building graph
    g = Digraph()
    
    # Introducing nodes
    nodes = []
    for i in range(7):
        nodes.append(Node(str(i)))
    # Adding them to the graph
    for node in nodes:
        g.add_node(node)
        

    # Adding edges to the graph and building the adjacency matrix
    tuples = [(0,1),(1,2),(2,3),(2,4),(3,4),(0,2),(1,5),(5,6),(2,6)]   
    weights = [1,2,1,4,2,5,8,9,2]
    w ={}
    for i,tup in enumerate(tuples):
        g.add_edge(WeightedEdge(nodes[tup[0]],nodes[tup[1]],weights[i]))
        w[(nodes[tup[0]],nodes[tup[1]])] = weights[i]
    source = g.nodes[0]
    print(f'source: {source}')
    parents,distances = dijkstra(g,source)
    
    for node in nodes:        
        print(f'distances[{node}] = {distances[node]}')
        print(f'p[{node}] = {parents[node]}')
    destination = g.nodes[6]
    shortest_path = SP(g,source,destination)
    [print(f'{node} ', end=" ") for node in shortest_path[::-1]]
   