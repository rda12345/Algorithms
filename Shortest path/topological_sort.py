#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topological sort

A topological sort of a directed graph is a linear ordering of its vertices
such that for every directed edge u → v, vertex u comes before v in the ordering.

- Works on a directed acyclic graph (DAG) (directed graph with no cycles)
- Multiple valid topological sorts may exist.

Applications:
    Shows dependency order — often used in task scheduling, building systems, and
    course prerequisite planning.
    
"""
from graph import *
from DFS import *


def topological_sort(graph,source):
    r = DFSResult()
    DFS_visit(graph,source,r)
    r.finished.reverse()
    return r.finished
    


def DFS_visit(g,v,r):
    for n in g.edges[v]:
        if n not in r.parents:
            r.parents[n] = v
            DFS_visit(g,n,r)
    
    r.finished.append(v)
    return


class DFSResult(object):
    def __init__(self):
        """
        Helper function for the topological sort
        
        Parameters:
            parents: dict, containing the nodes that have been visited by the DFS.
            finished: list, containing the nodes that have been visited in the
                        order they have been visited.
        """
        self.parents = {}
        self.finished = []
    

## Check
"""
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
        

    # Adding edges to the graph
    tuples = [(0,1),(1,2),(2,3),(2,4),(3,4),(0,2),(1,5),(5,6),(2,6)]    
    for tup in tuples:
        g.add_edge(Edge(nodes[tup[0]],nodes[tup[1]]))
        
    ordered_list = topological_sort(g,g.nodes[0]) 
    [print(i) for i in ordered_list]
"""    
            

    