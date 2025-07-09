import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh

def build_monomer_linear_rotor_hamiltonian(JM_list, B_const_K, dipole_terms):
	dim = len(JM_list)
	H = lil_matrix((dim, dim))
	for i, (J, M) in enumerate(JM_list):
		for j in range(i, dim):
			Jp, Mp = JM_list[j]
			if M != Mp:
				continue
			h_rot = B_const_K * J * (J + 1) if J == Jp else 0.0
			h_dip = dipole_terms.get((J, Jp, M), 0.0)
			val = h_rot + h_dip
			if abs(val) > 1e-12:
				H[i, j] = val
				if i != j:
					H[j, i] = val
	return H.tocsr()

def plot_sparsity(H, JM_list, save_path=None, dpi=600, max_labels=30, color='black'):
	"""
	Plot a high-quality sparsity pattern of the Hamiltonian matrix with clean aesthetics.

	Parameters:
	- H : csr_matrix
		Sparse Hamiltonian matrix.
	- JM_list : ndarray
		Array of (J, M) quantum numbers defining basis states.
	- save_path : str or None
		If specified, saves the plot to the given path.
	- dpi : int
		Resolution for saved figure (default 600).
	- max_labels : int
		Max dimension for which tick labels (J, M) will be shown.
	- color : str
		Marker color for the plot (e.g., 'black', 'navy', '#1f77b4').
	"""
	#plt.style.use('seaborn-white')
	plt.style.use('seaborn-v0_8-whitegrid')  # if available
	dim = len(JM_list)

	fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=dpi)
	ax.spy(H, markersize=3.2, color=color, precision=1e-12)

	ax.set_title("Sparsity Pattern of the Hamiltonian Matrix", fontsize=14, fontweight='bold', pad=15)
	ax.set_xlabel("Ket state index  (J', M')", fontsize=12, labelpad=10)
	ax.set_ylabel("Bra state index  (J, M)", fontsize=12, labelpad=10)

	if dim <= max_labels:
		labels = [f"{int(J)},{int(M)}" for J, M in JM_list]
		ax.set_xticks(np.arange(dim))
		ax.set_xticklabels(labels, rotation=90, fontsize=7, family='monospace')
		ax.set_yticks(np.arange(dim))
		ax.set_yticklabels(labels, fontsize=7, family='monospace')
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	ax.tick_params(direction='in', top=True, right=True)
	ax.grid(False)
	plt.tight_layout()

	if save_path:
		fig.savefig(save_path, bbox_inches='tight', dpi=dpi, transparent=True)
		print(f"Sparsity pattern saved to: {save_path}")
	else:
		plt.show()

def plot_sparsity(H, JM_list, save_path=None, dpi=600, max_labels=30, color='black'):
	"""
	Plot a high-quality sparsity pattern of the Hamiltonian matrix.

	Parameters:
	- H : csr_matrix
		Sparse Hamiltonian matrix.
	- JM_list : ndarray
		Array of (J, M) quantum numbers.
	- save_path : str or None
		If provided, the plot is saved to this path.
	- dpi : int
		Resolution of saved plot.
	- max_labels : int
		Max matrix size for which tick labels are shown.
	- color : str
		Marker color in the sparsity plot.
	"""

	# Fallback style (safe on all systems)
	plt.rcParams.update({
		"font.size": 10,
		"axes.titlesize": 13,
		"axes.labelsize": 11,
		"xtick.labelsize": 8,
		"ytick.labelsize": 8,
		"xtick.direction": 'in',
		"ytick.direction": 'in',
		"axes.edgecolor": 'gray',
		"axes.linewidth": 0.8,
	})

	dim = len(JM_list)
	fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=dpi)

	ax.spy(H, markersize=3.2, color=color, precision=1e-12)
	ax.set_title("Hamiltonian Matrix Sparsity", pad=15, fontweight='bold')
	ax.set_xlabel("Ket index (J', M')", labelpad=8)
	ax.set_ylabel("Bra index (J, M)", labelpad=8)

	if dim <= max_labels:
		labels = [f"{int(J)},{int(M)}" for J, M in JM_list]
		ax.set_xticks(np.arange(dim))
		ax.set_yticks(np.arange(dim))
		ax.set_xticklabels(labels, rotation=90, fontsize=7, family='monospace')
		ax.set_yticklabels(labels, fontsize=7, family='monospace')
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	ax.tick_params(top=True, right=True)
	ax.grid(False)
	fig.patch.set_facecolor('white')  # Ensure clean background
	plt.tight_layout()

	if save_path:
		fig.savefig(save_path, bbox_inches='tight', dpi=dpi, transparent=False)
		print(f"Sparsity plot saved to: {save_path}")
	else:
		plt.show()

def diagonalize(H, num_eig=6):
	dim = H.shape[0]
	if dim <= 300:
		return np.sort(eigh(H.toarray())[0])
	else:
		return np.sort(eigsh(H, k=min(num_eig, dim - 2), which='SA')[0])

def compute_sorted_eigenvalues_and_eigenvectors(H_rot):
	"""
	Computes and sorts the eigenvalues and eigenvectors of the rotational Hamiltonian matrix.

	Parameters:
	- H_rot (ndarray): Rotational Hamiltonian matrix (NxN).
	- scaling_factor (float): Scaling factor for eigenvalues (unit conversion).

	Returns:
	- eigenvalue_matrix (ndarray): Nx2 matrix with sorted eigenvalues and their scaled versions.
	- sorted_eigenvectors (ndarray): NxN matrix of sorted eigenvectors.
	"""
	# Compute eigenvalues and eigenvectors
	eigenvalues, eigenvectors = eigh(H_rot)

	# Sort eigenvalues and eigenvectors
	sorted_indices = np.argsort(eigenvalues)
	sorted_eigenvalues = eigenvalues[sorted_indices]
	sorted_eigenvectors = eigenvectors[:, sorted_indices]

	# Create a matrix with eigenvalues and their scaled versions
	#eigenvalue_matrix = np.column_stack((sorted_eigenvalues, sorted_eigenvalues / scaling_factor))

	#return eigenvalue_matrix, sorted_eigenvectors
	return sorted_eigenvalues, sorted_eigenvectors


