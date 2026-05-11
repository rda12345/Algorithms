"""
The Lloyd procedure for k-means clustering

Given n, d-dimensional, data points, the algorithm finds the position of k "centers", associated with
k clusters. The clusters are sets of disjoint points which are closer to their associate
center than any other center. Since there are a finite number of partitions (k^n) the procedure
is guaranteed to converge after a finite number of iterations.

Complexity: Each iteration has a complexity of O(n*d*k). Therefore, after T
            iterations, the total running time is O(T*n*d*k)
"""
import numpy as np
import matplotlib.pyplot as plt

def distance(a, b):
    """
    Euclidean distance between two points.
    """
    return np.linalg.norm(a - b)

#TODO: code the algorithm
def lloyd_clustering(data: np.ndarray, k:int, max_iters:int=100):
    """
    Returns the position of k "centers" associated with the k-means clustering algorithm.

    Args:
        data: np.array, the columns of data constitute the data points
        k: int, the number of clusters
    """
    n = data.shape[1]   # number of data points
    # initialize centers by sampling k data points
    indices = np.random.choice(n, k, replace=False)
    centers = [data[:, i].copy() for i in indices]

    # assign points to clusters
    cluster_dict = {i: np.random.choice(k) for i in range(n)}

    for iteration in range(max_iters):
        converged = True
        for i in range(n):
            distances = [distance(data[:, i], center) for center in centers]
            cluster_idx = np.argmin(distances)
            if cluster_idx != cluster_dict[i]:
                converged = False
                cluster_dict[i] = cluster_idx

        # stop if there is no change
        if converged:
            return cluster_dict, centers, iteration

        # recompute centers as centroids
        for i in range(k):
            cluster = [data[:, j] for j, c in cluster_dict.items() if c == i]
            centers[i] = np.mean(np.array(cluster), axis=0)

if __name__ == "__main__":
    X = np.random.rand(2, 100)
    k = 3
    cluster_dict, centers, iterations = lloyd_clustering(data=X, k=k)
    print(f"Algorithm converged after {iterations} iterations.")

    colors = plt.cm.tab10.colors
    for point_idx, cluster_idx in cluster_dict.items():
        plt.scatter(X[0, point_idx], X[1, point_idx], color=colors[cluster_idx], alpha=0.6, s=30)
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], color=colors[i], marker="o", s=200, edgecolors="black", linewidths=1)
    plt.title(f"K-Means Clustering (k={k}), converged after {iterations} iterations")
    plt.show()
    

    

