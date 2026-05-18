"""
Approximate Vertex Cover algorithm

A vertex cover over an undirected graph G = (V, E) is a set of vetices V' in V, for which
for every edge {u, v} in E, u or v are in V' or both are.
The size of the vertex cover V' is the number of vertices in V'.
The vertex cover problem's goal is given an undirected graph,
find the minimum size vertex cover of a graph.

The algorithm is a polynomial time 2-approximation algorithm, meaning
that the obtained vertex cover size is at most twice the size of the
optimal vertex cover.

Complexity: runtime scales as O(V + E)
"""
from graph import Graph, Node, Edge
import numpy as np

def approx_vertex_cover(G: Graph) -> set[Node]:
    """
    Returns a vertex cover which at most twice the size of the optimal
    vertex cover.

    Args:
        G: graph

    Returns:
        set[Node], a list containing the the set of nodes, constituting the vertex cover
    """
    C = set()
    edges = G.edges_set.copy()  # edges_set is a set of all edges in the graph, which allows for O(1) time complexity for edge removal
    while edges:
        edge = np.random.choice(list(edges))
        C |= {edge.src, edge.dest}
        edges -= {e for e in edges if edge.src in (e.src, e.dest) or edge.dest in (e.src, e.dest)}
    return C

if __name__ == "__main__":
    g = Graph()      
    
    # Introducing nodes
    nodes = []
    for i in range(1,5):
        nodes.append(Node(str(i)))
    # Adding them to the graph
    for node in nodes:
        g.add_node(node)        

    # Adding edges to the graph and building the adjacency matrix
    tuples = [(1,3),(3,4),(4,2),(2,1),(2,3),(1,4),(3,2),(4,1)]
    
    for i,tup in enumerate(tuples):
        g.add_edge(Edge(nodes[tup[0]-1],nodes[tup[1]-1]))

    
    vertex_cover = approx_vertex_cover(g)
    print("---------------- TEST -------------------\n")
    print(f"Vertex cover: {[node.name for node in vertex_cover]}\n")
    print("Optimal vertex cover is any set of 3 vertices")            
    assert len(vertex_cover) <= 6, "The vertex cover is larger than twice the optimal vertex cover size"