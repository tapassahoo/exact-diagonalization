import numpy as np
from numpy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh

def compute_eigensystem(H, num_eig=6, return_vectors=True):
	"""
	Computes sorted eigenvalues (and optionally eigenvectors) of a Hamiltonian matrix.

	Parameters:
	- H (ndarray or csr_matrix): Hamiltonian matrix (can be dense or sparse).
	- num_eig (int): Number of eigenvalues to compute for large matrices (used if sparse).
	- return_vectors (bool): Whether to return eigenvectors.

	Returns:
	- eigenvalues (ndarray): Sorted eigenvalues.
	- eigenvectors (ndarray or None): Corresponding eigenvectors if return_vectors is True.
	"""
	# Convert sparse matrix to dense if needed
	if issparse(H):
		H = H.toarray()

	dim = H.shape[0]

	# Choose eigensolver based on matrix size
	if dim <= 300:
		eigenvalues, eigenvectors = eigh(H)
	else:
		if return_vectors:
			eigenvalues, eigenvectors = eigsh(H, k=min(num_eig, dim - 2), which='SA')
		else:
			eigenvalues = eigsh(H, k=min(num_eig, dim - 2), which='SA', return_eigenvectors=False)
			eigenvectors = None

	# Sort eigenvalues and eigenvectors
	sorted_indices = np.argsort(eigenvalues)
	eigenvalues = eigenvalues[sorted_indices]
	eigenvectors = eigenvectors[:, sorted_indices] if return_vectors and eigenvectors is not None else None

	return eigenvalues, eigenvectors

