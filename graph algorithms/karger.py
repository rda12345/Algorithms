"""
Karger's Algorithm for Minimum Cut

The algorithm is based on the idea of contracting edges in the graph, until only two vertices remain.
The edges between these two vertices represent a cut of the original graph.
By repeating this process multiple times, we can find a minimum cut
with propability at least >=2/(|V|*(|V|-1)), where |V| is the number of vertices in the graph.
In comparison a naive randomized algorithm, which picks a random cut,
has a probability of 2^{-Omega(|V|)} to find the minimum cut (we can think of the problem as choosing one
of the two sets n times).
Repeating Karger's algorithm O(|V|^2) time and taking the minimum cut gives a O(1) probability to find
minimum cut.

Complexity: With ... data structure each iteration takes O(|E|) time, therefore,
            to achieve unit probability requires O(|E|*|V|^2).
"""
#TODO: code the algorithm, test it, and analyze its complexity.
#TODO: understand which data structures make the algorithm much faster
#TODO: complete the comnplexity analysis
import numpy as np
from graph import Graph, Node, Edge

def karger(graph: Graph, iterations: int=100) -> list[Edge]:
    """
    Returns a minimum cut of the input graph, with high probability.

    Args:
        graph: dict, adjacency list representation of the graph
        iterations: int, number of iterations to run the algorithm
    
    Returns:
        cut: list of edges, the minimum cut found by the algorithm
    """
    E = len(graph.edges)
    for _ in range(iterations):   
        edges = np.random(graph.edges)     # permute the edges of the graph 
        edges.pop()     # remove the last edge in the graph
        V = [[node] for node in graph.nodes]
        for edge in edges:
            V = contract_edge(edge, V)     # outputs a list of lists, including combined vertices
    
    # evaluate how many 
        

def contract_edge(edge: Edge, V: list[list]) -> list[list]:
    # TODO: combine the src and dest nodes in the list
                                        


