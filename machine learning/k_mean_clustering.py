"""
The Lloyd procedure for k-means clustering

Given n, d-dimensional, data points, the algorithm finds the position of k "centers", associated with
k clusters. The clusters are sets of disjoint points which are closer to their associate
center than any other center. Since there are a finite number of partitions (k^n) the procedure
is guaranteed to converge after a finite number of iterations. .

Complexity: Each iteration has a complexity of O(n*d*k). Therefore, after T
            iterations, the total running time is O(T*n*d*k)
"""


def distance(a, b):
    """
    Euclidean distance between two points.
    """
    return np.linalg.norm(a - b)

#TODO: code the algorithm
def lloyd_clustering(data, k):
    """
    Returns the position of k "centers" associated with the k-means clustering algorithm.

    Args:
        data: np.array, the columns of data constitute the data points
        k: int, the number of clusters
    """
    d = data.shape[0]   # dimension of a data point
    n = data.shape[1]   # number of data points
    # initialize centers randomly
    centers = [np.random.rand(d) for _ in range(k)]

    # assign points to clusters
    clusters = [{i: None} for _ in range(k)]
    converged = True
    for i in range(n):
        center = centers[i]
        distances = distance(data, center)
        cluster_idx = np.argmin(distances)
        if cluster_idx != clusters[i]:
            converged = False
            clusters[i] = np.argmin(distances)

    # stop if there is no change
    if converged:
        return clusters
    # recompute centers as centroids
    for i in range(k):

        centroid =

