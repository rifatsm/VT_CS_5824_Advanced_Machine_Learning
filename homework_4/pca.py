import numpy as np

def pca(X):
    """
    Run principal component analysis.

    Parameters
    ----------
    X : array_like
        The dataset to be used for computing PCA. It has dimensions (m x n)
        where m is the number of examples (observations) and n is
        the number of features.

    Returns
    -------
    U : array_like
        The eigenvectors, representing the computed principal components
        of X. U has dimensions (n x n) where each column is a single
        principal component.

    S : array_like
        A vector of size n, contaning the singular values for each
        principal component. Note this is the diagonal of the matrix we
        mentioned in class.

    Instructions
    ------------
    You should first compute the covariance matrix. Then, you
    should use the "svd" function to compute the eigenvectors
    and eigenvalues of the covariance matrix.

    Notes
    -----
    When computing the covariance matrix, remember to divide by m (the
    number of examples).
    """
    # Useful values
    m, n = X.shape
    U = np.zeros(n)
    S = np.zeros(n)

    # You need to return U, S correctly.
    # ====================== YOUR CODE HERE ======================

    Sigma = (1/m)*np.dot(X.T,X)
    U,S,V = np.linalg.svd(Sigma)
    
    # ============================================================
    return U, S

def project_data(X, U, K):
    """
    Computes the reduced data representation when projecting only
    on to the top K eigenvectors.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n). The dataset is assumed to be
        normalized.

    U : array_like
        The computed eigenvectors using PCA. This is a matrix of
        shape (n x n). Each column in the matrix represents a single
        eigenvector (or a single principal component).

    K : int
        Number of dimensions to project onto. Must be smaller than n.

    Returns
    -------
    Z : array_like
        The projects of the dataset onto the top K eigenvectors.
        This will be a matrix of shape (m x k).

    Instructions
    ------------
    Compute the projection of the data using only the top K
    eigenvectors in U (first K columns).
    """
    # You need to return Z correctly.
    # ====================== YOUR CODE HERE ======================
    m, n = X.shape
    Z = np.zeros((m, K))
    
#     print("U.shape: ", U.shape)
    Ureduce = U[:, :K]
    
#     print("Ureduce.shape: ", Ureduce.shape)
#     print("X.shape: ", X.shape)
    
    Z = np.dot(Ureduce.T,X.T).T
    
#     print("Z.shape: ", Z.shape)    
    # =============================================================
    return Z

def recover_data(Z, U, K):
    """
    Recovers an approximation of the original data when using the
    projected data.

    Parameters
    ----------
    Z : array_like
        The reduced data after applying PCA. This is a matrix
        of shape (m x K).

    U : array_like
        The eigenvectors (principal components) computed by PCA.
        This is a matrix of shape (n x n) where each column represents
        a single eigenvector.

    K : int
        The number of principal components retained
        (should be less than n).

    Returns
    -------
    X_rec : array_like
        The recovered data after transformation back to the original
        dataset space. This is a matrix of shape (m x n), where m is
        the number of examples and n is the dimensions (number of
        features) of original datatset.

    Instructions
    ------------
    Compute the approximation of the data by projecting back
    onto the original space using the top K eigenvectors in U.
    """
    # You need to return X_rec correctly.
    # ====================== YOUR CODE HERE ======================
    m = Z.shape[0]
    n = U.shape[0]
    X_rec = np.zeros((m, n))
    
    Ureduce = U[:, :K]

    X_rec = np.dot(Ureduce, Z.T).T
    
    # =============================================================
    return X_rec
