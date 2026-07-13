

import numpy as np
from numpy.linalg import eig


def modal_controllability(A_norm):
    
    if A_norm.ndim != 2 or A_norm.shape[0] != A_norm.shape[1]:
        raise ValueError(
            f"A_norm must be a square 2D array. Got shape: {A_norm.shape}."
        )

    n_nodes = A_norm.shape[0]
    eigenvalues, eigenvectors = eig(A_norm)

   
    # Ensure complex magnitude is used for directed (asymmetric) matrices
    modal_weights = 1.0 - np.abs(eigenvalues) ** 2  

   
    mc = np.sum(modal_weights[np.newaxis, :] * (np.abs(eigenvectors) ** 2), axis=1)

    return np.real(mc)


def average_controllability(A_norm):
    
    if A_norm.ndim != 2 or A_norm.shape[0] != A_norm.shape[1]:
        raise ValueError(
            f"A_norm must be a square 2D array. Got shape: {A_norm.shape}."
        )

    eigenvalues, eigenvectors = eig(A_norm)
    # Ensure complex magnitude is used for directed (asymmetric) matrices
    denom = 1.0 - np.abs(eigenvalues) ** 2
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    ac = np.sum((1.0 / denom)[np.newaxis, :] * (np.abs(eigenvectors) ** 2), axis=1)

    return np.real(ac)


def rank_facilitator_nodes(A_norm, top_k=10):
    
    mc = modal_controllability(A_norm)
    ranked_idx = np.argsort(mc)[::-1]  
    return ranked_idx[:top_k], mc[ranked_idx[:top_k]]
