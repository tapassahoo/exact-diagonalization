import numpy as np
from termcolor import colored
from scipy.sparse import issparse
from pkg_utils.utils import whoami
from pkg_utils.config import *

def debug_eigenvalues_eigenvectors(H_rot, eigenvalues, eigenvectors, tol=1e-10, log_file=None):
	"""
	Validates eigenvalues and eigenvectors from a Hamiltonian matrix.

	Parameters:
	- H_rot: np.ndarray or scipy sparse matrix
		The Hamiltonian matrix (real symmetric or Hermitian).
	- eigenvalues: np.ndarray
		Array of eigenvalues (assumed sorted).
	- eigenvectors: np.ndarray
		Array of corresponding eigenvectors (columns are eigenvectors).
	- tol: float
		Tolerance for floating-point comparisons.
	- log_file: str or None
		If provided, logs the output to the specified file.
	"""
	def log(msg):
		print(msg)
		if log_file:
			with open(log_file, 'a') as f:
				f.write(msg + '\n')

	log("\n[DEBUG] Eigenvalue and Eigenvector Validation")

	# Convert sparse to dense if necessary
	H_dense = H_rot.toarray() if hasattr(H_rot, "toarray") else H_rot

	# Matrix size and number of eigenpairs
	N = H_dense.shape[0]
	k = eigenvectors.shape[1]

	# 1. Hermiticity check
	is_hermitian = np.allclose(H_dense, H_dense.T.conj(), atol=tol)
	log("[INFO] Hamiltonian is Hermitian." if is_hermitian else "[ERROR] Hamiltonian is NOT Hermitian.")
	assert is_hermitian, "[ERROR] Hamiltonian must be Hermitian."

	# 2. Eigenvalue sorting
	is_sorted = np.all(np.diff(eigenvalues) >= -tol)
	log("[INFO] Eigenvalues are sorted." if is_sorted else "[WARNING] Eigenvalues are not sorted.")
	assert is_sorted, "[ERROR] Eigenvalues must be sorted in ascending order."

	# 3. Orthonormality check: V†V ≈ I_k
	if eigenvectors.shape[0] == eigenvectors.shape[1]:  # N ≥ k
		gram_matrix = np.einsum("ij,kj->ik", eigenvectors.conj(), eigenvectors)
		identity_k = np.eye(k)
		is_orthonormal = np.allclose(gram_matrix, identity_k, atol=tol)
		log("[INFO] Eigenvectors are orthonormal." if is_orthonormal else "[ERROR] Eigenvectors are not orthonormal.")
		assert is_orthonormal, "[ERROR] Eigenvectors must be orthonormal."

		# 4. Check eigenvalue residuals: H v_i ≈ λ_i v_i
		Hv = H_dense @ eigenvectors
		Lv = eigenvectors @ np.diag(eigenvalues)
		residuals = Hv - Lv
		max_residual = np.max(np.abs(residuals))
		log(f"[INFO] Maximum residual ‖Hv - λv‖ = {max_residual:.2e}")
		#assert max_residual < tol, "[ERROR] Residual norm is too large."
	else:
		log(f"[WARNING] Skipping orthonormality check: eigenvectors shape {eigenvectors.shape} not suitable.")

	# 5. Full spectral decomposition: H ≈ V Λ V† (only when all eigenpairs are available)
	"""
	if N == k:
		H_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
		is_exact = np.allclose(H_dense, H_reconstructed, atol=1e-10)
		#H_reconstructed = np.einsum("ij,j,jk->ik", eigenvectors, eigenvalues, eigenvectors.conj().T)
		#is_exact = np.allclose(H_dense, H_reconstructed, atol=tol)
		log("[INFO] Spectral decomposition is accurate." if is_exact else "[ERROR] Reconstructed matrix mismatch.")
		assert is_exact, "[ERROR] Reconstructed Hamiltonian does not match original."
	else:
		log(f"[WARNING] Skipping spectral reconstruction: only {k} of {N} eigenpairs available.")
	"""

	# 6. Complex eigenvector warning
	if np.iscomplexobj(eigenvectors):
		log("[WARNING] Eigenvectors contain complex components.")

	log("[SUCCESS] All eigenpair validations completed.\n")


def debug_dipole_terms(dipole_terms, JM_list, tol=1e-12, debug=False):
	"""
	Debug and validate dipole matrix elements.
	"""

	if debug:
		print("\n===== DEBUGGING DIPOLE TERMS =====\n")

		JM_set = {(int(J), int(M)) for J, M in JM_list}

		# 1. Print all elements (sorted)
		print("Matrix Elements:")
		for key in sorted(dipole_terms.keys()):
			J, Jp, M = key
			val = dipole_terms[key]
			print(f"<{J},{M}|cos(theta)|{Jp},{M}> = {val:.8f}")

		# 2. Check selection rules
		print("\nChecking Selection Rules:")
		for (J, Jp, M), val in dipole_terms.items():
			if abs(Jp - J) != 1:
				print(f"Violation: delta J != 1 at {(J, Jp, M)}")
			if (J, M) not in JM_set or (Jp, M) not in JM_set:
				print(f"Violation: state not in basis {(J, Jp, M)}")

		print("Selection rule check complete")

		# 3. Check symmetry (Hermiticity)
		print("\nChecking Symmetry:")
		for (J, Jp, M), val in dipole_terms.items():
			if (Jp, J, M) in dipole_terms:
				val2 = dipole_terms[(Jp, J, M)]
				if abs(val - val2) > tol:
					print(f"Mismatch: ({J},{Jp},{M}) != ({Jp},{J},{M})")

		print("Symmetry check complete")

		# 4. Check missing expected elements
		print("\nChecking Missing Elements:")
		for (J, M) in JM_set:
			for dJ in (-1, 1):
				Jp = J + dJ
				if (Jp, M) in JM_set:
					if (J, Jp, M) not in dipole_terms:
						print(f"Missing element: ({J},{Jp},{M})")

		print("Missing element check complete")

		nonzero_count = sum(1 for val in dipole_terms.values() if abs(val) > tol)

		print("\n===== SUMMARY =====")
		print("Total stored elements:", len(dipole_terms))
		print("Total nonzero elements:", nonzero_count)
		print("=================================\n")
