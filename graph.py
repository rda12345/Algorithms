#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file introduces a Digraph and Graph class utilizing object oriented programming.
The constructtion follows chapter 17 of "Introduction to Computation and programming using Python".
The Digraph and Graph classes are built from Node and Edge classes. 
"""

class Node(object):

    def __init__(self,name):
        '''
        name: str, the name of the node.
        '''
        self.name = name
    
    def get_name(self):
        ''' Returns the name of the node.'''
        return self.name
    def __str__(self):
        return self.name


class Edge(object):
    
    def __init__(self,src,dest):
        '''
        The edge connects the source node to the destination node.
        
        src: node, name of the source node.
        dest: node, name of the destination node.
        '''
        self.src = src
        self.dest = dest
    

    # Getter methods for the source and destination nodes
    def get_source(self):
        return self.src
    
    def get_destination(self):
        return self.dest
    
    def __str__(self):
        return self.src.get_name() + '->' + self.dest.get_name()
    
class WeightedEdge(Edge):
    def __init__(self,src,dest,weight = 1.0):
        ''' src: node, source node
            dest: node, destination node
            weight: float, between 0 and 1.0
        '''
        self.src = src 
        self.dest = dest
        self.weight = weight
    
    def get_weight(self):
        return self.weight 
    
    def __str__(self):
        return self.src.get_name() + '->(' + str(self.weight)+')' + self.dest.get_name()

class Digraph(object):
    # nodes is a list of nodes in the graph
    # edges us a dictionary mapping each node to a list of its children nodes.
    
    def __init__(self):
        self.nodes = []
        self.edges = {}
    
    def add_node(self,node):
        if node in self.nodes:
            raise ValueError('Duplicate node')
        else:
            self.nodes.append(node)
            self.edges[node] = []
            
    
    def add_edge(self,edge):
        src = edge.get_source()
        dest = edge.get_destination()
         

        if not (src in self.nodes and dest in self.nodes):
            raise ValueError('Node not in the graph')
        self.edges[src].append(dest)
    
    def children_of(self,node):
        return self.edges[node]
    
    def __str__(self):
        result = ''
        for src in self.nodes:
            if len(self.edges[src])>0:
                for child in self.edges[src]:
                    result = result + str(src) + '->' + str(child) + '\n'
        return result[:-1] # Omits the last empty line.       

class Graph(Digraph):    
    def add_edges(self,edge):
        Digraph.add_edge(edge)
        rev = Digraph.add_edge(Edge(edge.get_destination(),edge.get_source()))
        Digraph.add_edge(rev)
        
        
