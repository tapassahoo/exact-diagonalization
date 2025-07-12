import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from termcolor import colored
from pkg_utils.utils import whoami
from pkg_utils.config import *

def rotational_energy_levels(B, J_max=10):
	"""
	Computes and displays the rotational energy levels of a rigid rotor.
	
	Parameters:
	- B (float): Rotational constant in cm⁻¹.
	- J_max (int): Maximum rotational quantum number to compute.
	
	Returns:
	- energies (dict): Dictionary with J values as keys and energy in cm⁻¹ as values.
	"""
	J_values = np.arange(0, J_max + 1)  # Rotational quantum numbers J = 0, 1, 2, ...
	energies = {J: B * J * (J + 1) for J in J_values}  # Energy formula E_J = B * J * (J + 1)
	
	# Display results
	print(colored("\nRotational energy levels of a rigid rotor", HEADER_COLOR, attrs=['bold', 'underline']))
	print(f"\n{'J':<5}{'Energy (cm^-1)':>15}")
	print("=" * 20)
	for J, E in energies.items():
		print(f"{J:<5}{E:>15.2f}")
	
	return energies

def plot_rotational_levels(
	energies: dict,
	show_degeneracy: bool = True,
	title_suffix: str = "",
	save_path: str = None
):
	""" 
	Plot rotational energy levels of a rigid rotor using LaTeX labels.

	Parameters:
		energies (dict): Keys are J values, values are energies (in cm⁻¹).
		show_degeneracy (bool): Whether to annotate 2J+1 degeneracy.
		title_suffix (str): Extra title text (e.g., ' (Field-Free)', ' for HF').
		save_path (str or None): If provided, saves the plot to this path.
	"""
	J_values = list(energies.keys())
	energy_values = list(energies.values())

	plt.figure(figsize=(10, 6))
	plt.vlines(J_values, 0, energy_values, color='royalblue', linewidth=2)
	plt.scatter(J_values, energy_values, color='crimson', s=80, zorder=3, label="Energy Levels")

	for J, E in energies.items():
		label = f"{E:.2f} " + r"$\mathrm{cm}^{-1}$"
		if show_degeneracy:
			label += f"\n({2*J + 1}×)"
		plt.text(J, E + max(energy_values) * 0.03, label,
				 ha='center', va='bottom', fontsize=9, color='black')

	plt.xticks(J_values, fontsize=10)
	plt.yticks(fontsize=10)
	plt.xlabel(r"Rotational Quantum Number $J$", fontsize=12)
	plt.ylabel(r"Rotational Energy $E_J$ (in $\mathrm{cm}^{-1}$)", fontsize=12)
	plt.title(r"Rotational Energy Levels of a Rigid Rotor" + title_suffix, fontsize=14, weight='bold')
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.tight_layout()
	plt.legend()

	# Save or show
	if save_path:
		plt.savefig(save_path, dpi=300)
		print(f"[Info] Plot saved to: {save_path}")
	else:
		plt.show()

if False:
	def plot_rotational_levels(energies):
		""" 
		Plots the rotational energy levels of a rigid rotor with enhanced aesthetics.
		
		Parameters:
		- energies (dict): Dictionary where keys are rotational quantum numbers (J) and
		  values are the corresponding energy levels in cm^-1.
		"""
		J_values = list(energies.keys())
		energy_values = list(energies.values())

		max_energy = max(energy_values)
		offset = max_energy * 0.04  # space for annotation

		plt.figure(figsize=(12, 6))
		
		# Plot vertical energy levels as steps
		for J in J_values:
			plt.hlines(energy_values[J], J - 0.3, J + 0.3, colors='teal', linewidth=3)
			plt.text(J, energy_values[J] + offset, f"{energy_values[J]:.2f} cm^-1", 
					 ha='center', va='bottom', fontsize=10, color='dimgray')

		# Formatting axes
		plt.xticks(J_values, fontsize=10)
		plt.yticks(fontsize=10)
		plt.xlabel("Rotational Quantum Number $J$", fontsize=12, weight='bold')
		plt.ylabel("Rotational Energy (cm^-1)", fontsize=12, weight='bold')
		plt.title("Rotational Energy Levels of a Rigid Rotor", fontsize=14, weight='bold')

		# Visual tweaks
		plt.grid(axis='y', linestyle='--', alpha=0.5)
		plt.xlim(min(J_values) - 1, max(J_values) + 1)
		plt.ylim(0, max_energy + 4 * offset)
		plt.tight_layout()
		plt.box(False)
		plt.show()


def build_monomer_linear_rotor_hamiltonian(JM_list, B_const, dipole_terms):
	dim = len(JM_list)
	H = lil_matrix((dim, dim))
	for i, (J, M) in enumerate(JM_list):
		for j in range(i, dim):
			Jp, Mp = JM_list[j]
			if M != Mp:
				continue
			h_rot = B_const * J * (J + 1) if J == Jp else 0.0
			h_dip = dipole_terms.get((J, Jp, M), 0.0)
			val = h_rot + h_dip
			if abs(val) > 1e-12:
				H[i, j] = val
				if i != j:
					H[j, i] = val
	return H.tocsr()

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

def display_rotational_energies(diagonal_elements, all_quantum_numbers, B_const_cm_inv):
	"""
	Displays the extracted diagonal elements as rotational energy levels.

	Parameters:
	- diagonal_elements (np.ndarray): Extracted diagonal elements representing energy levels.
	- all_quantum_numbers (np.ndarray): Array of quantum numbers, where each row represents a state
											  and the first column contains the J values (rotational quantum numbers).
	- B_const_cm_inv (float): The rotational constant (cm⁻¹), used to compute rotational energy levels.

	Returns:
	None
	"""
	print("\nRotational Energy Levels")
	print("=" * 80)
	print(f"{'Quantum State (J)':^25} {'BJ(J+1) (cm^-1)':^25} {'<JM|T|JM> (cm^-1)':^25}")
	print("=" * 80)

	# Extracting J values from the quantum numbers data
	J_values = all_quantum_numbers[:, 0]

	# Compute the rotational energy levels B * J(J+1)
	for J, energy in zip(J_values, diagonal_elements):
		# Calculate the theoretical energy level based on the B constant
		theoretical_energy = B_const_cm_inv * J * (J + 1)
		
		
		# Display the results
		print(f"{int(J):>12} {theoretical_energy:>32.6f} {energy:>26.6f}")

	print("=" * 80)
