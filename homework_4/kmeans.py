import numpy as np
import sys

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : array_like
        The dataset of size (m, n) where each row is a single example.
        That is, we have m examples each of n dimensions.

    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the the data dimension.

    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset X.

    Instructions
    ------------
    Go over every example, find its closest centroid, and store
    the index inside `idx` at the appropriate location.
    Concretely, idx[i] should contain the index of the centroid
    closest to example i. Hence, it should be a value in the
    range 0..K-1

    Note
    ----
    Compute the distance to find the corresponding centroids for each data.
    Tips: It is possible to encounter 0 when computing the distance.
          Remember that you can add a very small number to the np.sqrt
    """
    # Set K
    K = centroids.shape[0]

    # You need to return idx correctly.
    # ====================== YOUR CODE HERE ======================
#     print("X: ", X.shape) # (300,2)
#     print("centroids: ", centroids.shape) # (3, 2)
    M = X.shape[0]
    idx = np.zeros(M, dtype=np.uint16)
    eps = 1e-10

    X_new = np.sum(X*X, axis=1).reshape((M, 1))
    cen_new = np.sum(centroids * centroids, axis=1).reshape((1, K))
    temp = np.dot(X, centroids.T)
    dists = np.sqrt(X_new + cen_new - 2*temp + eps)
#     print("dists: ", dists.shape)

    idx = np.argmin(dists, axis=1)
    # =============================================================
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.

    Parameters
    ----------
    X : array_like
        The datset where each row is a single data point. That is, it
        is a matrix of size (m, n) where there are m datapoints each
        having n dimensions.

    idx : array_like
        A vector (size m) of centroid assignments (i.e. each entry in range [0 ... K-1])
        for each example.

    K : int
        Number of clusters

    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data
        points assigned to it.

    Instructions
    ------------
    Go over every centroid and compute mean of all points that
    belong to it. Concretely, the row vector centroids[i, :]
    should contain the mean of the data points assigned to
    cluster i.

    Note:
    -----
    Compute the new centroids.
    """
    # Useful variables
    m, n = X.shape
    centroids = np.zeros((K, n))
    # You need to return centroids correctly.
    # ====================== YOUR CODE HERE ======================
#     print("idices of idx where idx == 0: ", np.nonzero(idx == 0)) 
    for i in range(K):
        centroids[i] = np.mean(X[np.nonzero(idx == i)], axis=0)
    # =============================================================
    return centroids

def init_kmeans_centroids(X, K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x.

    Parameters
    ----------
    X : array_like
        The dataset of size (m x n).

    K : int
        The number of clusters.

    Returns
    -------
    centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).

    Instructions
    ------------
    You should set centroids to randomly chosen examples from the dataset X.
    """

    # You should return centroids correctly
    # ====================== YOUR CODE HERE ======================
    # Initialize the centroids to be random examples
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    # =============================================================
    return centroids
