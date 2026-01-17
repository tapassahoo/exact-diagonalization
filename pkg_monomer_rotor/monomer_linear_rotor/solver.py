import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import eigsh

def compute_eigensystem(H, num_eig=6, return_vectors=True, check_residual=False, verbose=True):
	"""
	Compute eigenvalues and eigenvectors of a Hamiltonian matrix.

	Parameters
	----------
	H : ndarray or sparse matrix
		The Hamiltonian matrix.
	num_eig : int
		Number of eigenvalues to compute (used for large or sparse matrices).
	return_vectors : bool
		Whether to return eigenvectors.
	check_residual : bool
		If True, computes and prints max residual ||Hv - λv||.
	verbose : bool
		If True, prints diagnostic messages.

	Returns
	-------
	eigenvalues : ndarray
	eigenvectors : ndarray or None
	"""
	# Convert sparse to dense if appropriate
	if issparse(H):
		dim = H.shape[0]
		if dim <= 6000:
			H = H.toarray()
		else:
			# Use sparse eigensolver
			if return_vectors:
				eigvals, eigvecs = eigsh(H, k=min(num_eig, dim - 2), which='SA')
			else:
				eigvals = eigsh(H, k=min(num_eig, dim - 2), which='SA', return_eigenvectors=False)
				eigvecs = None

			sorted_idx = np.argsort(eigvals)
			eigvals = eigvals[sorted_idx]
			if return_vectors and eigvecs is not None:
				eigvecs = eigvecs[:, sorted_idx]

			return eigvals, eigvecs


	# Ensure it's a dense array
	H = np.asarray(H, dtype=np.complex128)

	# Use dense eigensolver
	eigvals, eigvecs = eigh(H)

	# Sort
	sorted_idx = np.argsort(eigvals)
	eigvals = eigvals[sorted_idx]
	eigvecs = eigvecs[:, sorted_idx] if return_vectors else None

	# Residual check
	if check_residual and return_vectors:
		Hv = H @ eigvecs
		Lv = eigvecs @ np.diag(eigvals)
		residual = Hv - Lv
		max_res = np.max(np.abs(residual))
		if verbose:
			print(f"[INFO] Max residual ||Hv - λv|| = {max_res:.2e}")
		# Optionally add: assert max_res < 1e-10, "Residual too large!"

	return eigvals, eigvecs
