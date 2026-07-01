from matplotlib.colors import LogNorm
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator
from pkg_utils.utils import whoami
from pkg_utils.config import *


def rotational_energy_levels(B, J_max=10, return_dict=False, display=False):
	"""
	Compute rotational energy levels of a rigid rotor.

	Parameters
	----------
	B : float
		Rotational constant (cm⁻¹)

	J_max : int, optional
		Maximum rotational quantum number

	return_dict : bool, optional
		If True, return a dictionary {J: E_J}; otherwise return arrays

	display : bool, optional
		If True, print formatted table

	Returns
	-------
	J : ndarray
		Rotational quantum numbers

	E : ndarray
		Energies in cm⁻¹

	or

	energies : dict
		Mapping {J: E_J} if return_dict=True
	"""

	import numpy as np

	J = np.arange(0, J_max + 1)
	E = B * J * (J + 1)

	# Display results
	if display:
		print(colored("\nRotational energy levels of a rigid rotor", HEADER_COLOR, attrs=['bold', 'underline']))
		print(f"\n{'J':<5}{'Energy (cm^-1)':>15}")
		print("=" * 20)
		for j, e in zip(J, E):
			print(f"{j:<5}{e:>15.6f}")

	if return_dict:
		return dict(zip(J, E))
	else:
		return J, E

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
	plt.scatter(
		J_values,
		energy_values,
		color='crimson',
		s=80,
		zorder=3,
		label="Energy Levels")

	for J, E in energies.items():
		label = f"{E:.2f} " + r"$\mathrm{cm}^{-1}$"
		if show_degeneracy:
			label += f"\n({2 * J + 1}×)"
		plt.text(J, E + max(energy_values) * 0.03, label,
				 ha='center', va='bottom', fontsize=9, color='black')

	plt.xticks(J_values, fontsize=10)
	plt.yticks(fontsize=10)
	plt.xlabel(r"Rotational Quantum Number $J$", fontsize=12)
	plt.ylabel(r"Rotational Energy $E_J$ (in $\mathrm{cm}^{-1}$)", fontsize=12)
	plt.title(r"Rotational Energy Levels of a Rigid Rotor" +
			  title_suffix, fontsize=14, weight='bold')
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
			plt.hlines(
				energy_values[J],
				J - 0.3,
				J + 0.3,
				colors='teal',
				linewidth=3)
			plt.text(J,
					 energy_values[J] + offset,
					 f"{energy_values[J]:.2f} cm^-1",
					 ha='center',
					 va='bottom',
					 fontsize=10,
					 color='dimgray')

		# Formatting axes
		plt.xticks(J_values, fontsize=10)
		plt.yticks(fontsize=10)
		plt.xlabel("Rotational Quantum Number $J$", fontsize=12, weight='bold')
		plt.ylabel("Rotational Energy (cm^-1)", fontsize=12, weight='bold')
		plt.title(
			"Rotational Energy Levels of a Rigid Rotor",
			fontsize=14,
			weight='bold')

		# Visual tweaks
		plt.grid(axis='y', linestyle='--', alpha=0.5)
		plt.xlim(min(J_values) - 1, max(J_values) + 1)
		plt.ylim(0, max_energy + 4 * offset)
		plt.tight_layout()
		plt.box(False)
		plt.show()


def build_monomer_linear_rotor_hamiltonian(
	JM_list, B_const, dipole_terms, tol=1e-12, debug=False
):
	"""
	Construct the Hamiltonian matrix for a linear rigid rotor
	in the |J, M> basis.

	Parameters
	----------
	JM_list : list of tuple
		List of (J, M) quantum numbers
	B_const : float
		Rotational constant
	dipole_terms : dict
		Dictionary of dipole matrix elements {(J, Jp, M): value}
	tol : float
		Numerical tolerance for zero
	debug : bool
		If True, print diagnostics

	Returns
	-------
	H : csr_matrix
		Sparse Hamiltonian matrix
	"""

	dim = len(JM_list)
	H = lil_matrix((dim, dim))

	nonzero_count = 0
	rot_count = 0
	dip_count = 0

	for i, (J, M) in enumerate(JM_list):
		for j in range(i, dim):
			Jp, Mp = JM_list[j]

			# Selection rule: M must be conserved
			if M != Mp:
				continue

			# Rotational term (diagonal)
			h_rot = B_const * J * (J + 1) if J == Jp else 0.0
			if abs(h_rot) > tol:
				rot_count += 1

			# Dipole term
			h_dip = dipole_terms.get((J, Jp, M), 0.0)
			if abs(h_dip) > tol:
				dip_count += 1

			val = h_rot + h_dip

			if abs(val) > tol:
				H[i, j] = val
				if i != j:
					H[j, i] = val
				nonzero_count += 1 if i == j else 2

				if debug:
					print(
						f"H[{i},{j}] <- (J,M)=({J},{M}) → (J',M')=({Jp},{Mp}) | "
						f"rot={h_rot:.3e}, dip={h_dip:.3e}, total={val:.3e}"
					)

	if debug:
		print("\n===== DEBUG SUMMARY =====")
		print(f"Matrix dimension		: {dim}")
		print(f"Total nonzero elements  : {nonzero_count}")
		print(f"Rotational contributions: {rot_count}")
		print(f"Dipole contributions	: {dip_count}")
		print("========================\n")

	return H.tocsr()


def plot_coupling_with_dJ_overlay(
	H,
	JM_list,
	dJ_values=(0, 1, -1, 2, -2),
	save_path=None,
	log_scale=True,
	cmap=None,
	overlay_color=None,
	overlay_alpha=0.9,
	max_labels=40,
):
	"""
	Plot |H_ij| with ΔJ selection-rule overlay.

	Parameters
	----------
	H : matrix (dense or sparse)
	JM_list : list of (J, M)
	dJ_values : tuple
		Allowed ΔJ values to overlay
	"""

	dim = H.shape[0]

	# Convert to dense if needed
	H_abs = np.abs(H.toarray() if hasattr(H, "toarray") else H)
	H_abs[H_abs == 0] = 1e-16  # avoid log issues

	J_vals = np.array([J for J, M in JM_list])

	# ------------------------------------------------------------------
	# Plot base heatmap
	# ------------------------------------------------------------------
	fig, ax = plt.subplots(figsize=(8, 8))

	norm = LogNorm(vmin=H_abs.min(), vmax=H_abs.max()) if log_scale else None

	im = ax.imshow(
		H_abs,
		origin="upper",
		cmap=cmap,
		norm=norm,
		aspect="equal",
	)

	# ------------------------------------------------------------------
	# Overlay ΔJ bands
	# ------------------------------------------------------------------
	for dJ in dJ_values:
		mask = (J_vals[:, None] - J_vals[None, :]) == dJ

		# Extract indices where condition holds
		y, x = np.where(mask)

		ax.scatter(
			x,
			y,
			s=1.0,
			color=overlay_color,
			alpha=overlay_alpha,
			label=fr"$\Delta J = {dJ}$" if dJ == dJ_values[0] else None,
		)

	# ------------------------------------------------------------------
	# Colorbar
	# ------------------------------------------------------------------
	cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(r"$\log_{10}(|H_{ij}|)$", fontsize=12)

	# ------------------------------------------------------------------
	# Labels
	# ------------------------------------------------------------------
	#ax.set_title("Hamiltonian Coupling with $\Delta J$ Selection Rules", fontweight="bold", pad=15)

	# Tick labels
	if dim <= max_labels:
		ticks = np.arange(dim)

		def fmt(x):
			return str(int(x)) if x == int(x) else f"{x:.1f}"


		def fmt_M(x):
			if x > 0:
				return f"+{int(x)}" if x == int(x) else f"+{x:.1f}"
			elif x == 0:
				return rf"\,{int(x)}"   # no + sign
			else:
				return f"{int(x)}" if x == int(x) else f"{x:.1f}"


		xlabels = [rf"$|{fmt(J)},{fmt_M(M)}\rangle$" for J, M in JM_list]
		ylabels = [rf"$\langle{fmt(J)},{fmt_M(M)}|$" for J, M in JM_list]

		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(xlabels, rotation=90, fontsize=12)
		ax.set_yticklabels(ylabels, fontsize=12)
		# Move x-axis labels to the top
		ax.xaxis.tick_top()
		ax.tick_params(direction="in", top=True, right=True)
	else:
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())

	# Styling
	ax.tick_params(direction="in", top=True, bottom=True, right=True)
	for spine in ax.spines.values():
		spine.set_linewidth(1.0)

	plt.tight_layout()

	# Save or show
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches="tight")
	else:
		plt.show()

	plt.close(fig)

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
	# eigenvalue_matrix = np.column_stack((sorted_eigenvalues, sorted_eigenvalues / scaling_factor))

	# return eigenvalue_matrix, sorted_eigenvectors
	return sorted_eigenvalues, sorted_eigenvectors


def display_rotational_energies(
		diagonal_elements,
		all_quantum_numbers,
		B_const_cm_inv):
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
