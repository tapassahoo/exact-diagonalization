#*************************************************************************************************#
#																								  #
# Computation of Eigenvalues and Eigenfunctions of a Linear Rotor Using Real Spherical Harmonics. #
#																								  #
# Developed by Dr. Tapas Sahoo																	  #
#																								  #
#-------------------------------------------------------------------------------------------------#
#																								  #
# Command for running the code:																	  #
#																								  #
# Example:																						  #
# python monomer_rotor_real_basis_diagonalization.py 10.0 2 spinless							  #
#																								  #
#-------------------------------------------------------------------------------------------------#
#																								  #
# Inputs:																						  #
# a) Potential potential_strength = potential_strength											  #
# b) Highest value of Angular quantum number = max_angular_momentum_quantum_number				  #
# c) Specification of spin isomer = spin_state													  #
#																								  #
# Outputs: Eigenvalues and eigenfunctions														  #
#																								  #
#*************************************************************************************************#

import argparse
import os
import inspect
import sys
import getpass
import socket
import platform
import math
import numpy as np
from numpy.linalg import eigh
import scipy
import scipy.constants as const
from scipy import linalg as LA
from scipy.sparse.linalg import eigs, eigsh
import cmath
from datetime import datetime
from termcolor import colored
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from contextlib import redirect_stdout
import io
from pkg_utils.utils import whoami
from pkg_utils.config import *

# Imports basis functions of rotors (linear and nonlinear rotors)
import pkg_basis_func_rotors.basis_func_rotors as bfunc

def parse_arguments():
	"""
	Parses command-line arguments for the computation of eigenvalues and eigenfunctions
	of a linear quantum rotor in an external orienting potential using a real spherical harmonics basis.

	Returns
	-------
	argparse.Namespace
		A namespace object containing:
		- potential_strength : float
		- max_angular_momentum_quantum_number : int
		- spin : str
		- dipole_moment : float or None
		- electric_field : float or None
		- output_dir : str
		- dry_run : bool
	"""
	parser = argparse.ArgumentParser(
		prog="monomer_rotor_real_basis_diagonalization.py",
		description=(
			"Performs exact diagonalization of a linear rotor Hamiltonian\n"
			"in an external orienting potential using a real spherical harmonics basis."
		),
		epilog="Developed by Dr. Tapas Sahoo — Quantum Molecular Dynamics Group"
	)

	parser.add_argument(
		"--potential-strength", type=float, default=None,
		help="Strength of the external orienting potential (in cm⁻¹)."
	)

	parser.add_argument(
		"max_angular_momentum_quantum_number", type=int,
		help="Maximum angular momentum quantum number ℓ_max used for basis truncation. Must be ≥ 0."
	)

	parser.add_argument(
		"spin", type=str, choices=["spinless", "ortho", "para"],
		help="Nuclear spin isomer type: 'spinless', 'ortho', or 'para'."
	)

	parser.add_argument(
		"--dipole-moment", type=float, default=None,
		help="Dipole moment of the rotor molecule (in Debye)."
	)

	parser.add_argument(
		"--electric-field", type=float, default=None,
		help="Electric field strength (in kV/cm)."
	)

	parser.add_argument(
		"--output-dir", type=str, default="output",
		help="Directory where output files will be saved (default: 'output')."
	)

	parser.add_argument(
		"--dry-run", action="store_true",
		help="If set, only prints computed settings without executing the main routine."
	)

	args = parser.parse_args()

	# Auto-calculate potential strength if not provided
	if args.potential_strength is None:
		if args.dipole_moment is not None and args.electric_field is not None:
			args.potential_strength = 0.0168 * args.dipole_moment * args.electric_field
		else:
			print("Error: You must provide either --potential-strength or both --dipole-moment and --electric-field.")
			sys.exit(1)

	return args

def show_simulation_details(
	potential_strength_cm_inv,
	max_angular_momentum_quantum_number,
	spin_state,
	theta_grid_count,
	phi_grid_count,
	dipole_moment_D=None,
	electric_field_kVcm=None,
	computed_muE_cm_inv=None
):
	"""
	Display simulation input details including dipole-field interaction info if available.
	"""
	now = datetime.now()
	date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
	user_name = getpass.getuser()
	cwd = os.getcwd()
	home_dir = os.path.expanduser("~")

	print(colored("*" * 80, SEPARATOR_COLOR))
	print(colored("Date and Time:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(date_time.ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")

	print(colored("File System Details:", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("User Name:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(user_name.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Home Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(home_dir.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Current Working Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(cwd.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Package Location (bfunc):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(bfunc.__file__.ljust(VALUE_WIDTH), VALUE_COLOR))
	print()

	print(colored("Simulation Parameters", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("ℓ_max (Angular Momentum):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{max_angular_momentum_quantum_number}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Spin State:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(spin_state.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("V(θ) Strength:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{potential_strength_cm_inv:.5f} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))

	if dipole_moment_D is not None:
		print(colored("Dipole Moment:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{dipole_moment_D:.4f} D".ljust(VALUE_WIDTH), VALUE_COLOR))
	if electric_field_kVcm is not None:
		print(colored("Electric Field:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{electric_field_kVcm:.4f} kV/cm".ljust(VALUE_WIDTH), VALUE_COLOR))
	if computed_muE_cm_inv is not None:
		print(colored("μ·E (Interaction Energy):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{computed_muE_cm_inv:.5f} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))
	print()

	print(colored("Grid Information", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Theta Grid Count:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{theta_grid_count}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Phi Grid Count:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{phi_grid_count}".ljust(VALUE_WIDTH), VALUE_COLOR))

def generate_filename(
	spin_state: str,
	max_angular_momentum_quantum_number: int,
	potential_strength: float,
	theta_grid_count: int,
	phi_grid_count: int,
	dipole_moment_D: Optional[float] = None,
	electric_field_kVcm: Optional[float] = None,
	prefix: Optional[str] = ""
) -> str:
	"""
	Generates a descriptive filename for a quantum rotor system.

	Parameters
	----------
	spin_state : str
		The spin isomer type ("spinless", "para", or "ortho").
	max_angular_momentum_quantum_number : int
		Maximum angular momentum quantum number (ℓ_max).
	potential_strength : float
		Orienting potential strength in Kelvin (used only if dipole-field interaction is not specified).
	theta_grid_count : int
		Number of θ grid points.
	phi_grid_count : int
		Number of φ grid points.
	dipole_moment_D : float, optional
		Dipole moment in Debye (include only if electric_field_kVcm is also provided).
	electric_field_kVcm : float, optional
		Electric field strength in kV/cm (include only if dipole_moment_D is also provided).
	prefix : str, optional
		Optional prefix or directory path.

	Returns
	-------
	str
		A clear and descriptive filename.
	"""

	filename = (
		f"{prefix}HCl_{spin_state}_isomer_"
		f"lmax_{max_angular_momentum_quantum_number}_"
	)

	if dipole_moment_D is not None and electric_field_kVcm is not None:
		filename += (
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{electric_field_kVcm:.2f}kVcm_"
		)
	else:
		filename += f"potential_{potential_strength:.2f}K_"

	filename += f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}"

	return filename

def compute_legendre_quadrature(theta_grid_count, phi_grid_count, display_legendre_quadrature):
	"""
	Computes Gaussian quadrature points and weights for Legendre polynomials,
	along with phi grid points for a specified number of theta and phi points.

	Parameters:
	- theta_grid_count (int): Number of theta points for Legendre quadrature.
	- phi_grid_count (int): Number of phi points for uniform grid.
	- display_legendre_quadrature = (bool): If True, prints the quadrature points and weights.

	Returns:
	- xGL (np.ndarray): Gaussian quadrature points.
	- wGL (np.ndarray): Corresponding weights.
	- phixiGridPts (np.ndarray): Uniformly spaced phi grid points.
	- dphixi (float): Phi grid spacing.


	Gauss-Legendre quadrature points are given in https://en.wikipedia.org/wiki/Gaussian_quadrature
	"""

	# Optionally print the quadrature points and weights
	if display_legendre_quadrature:
		# Print separator line
		print()
		print(colored("*" * 80, SEPARATOR_COLOR))
		print( colored( "Gauss-Legendre quadrature points are given in ", DEBUG_COLOR) + colored( "https://en.wikipedia.org/wiki/Gaussian_quadrature", INFO_COLOR, attrs=[ 'bold', 'underline']) + "\n")

		# Iterate through the different values of itheta_grid_count
		for itheta_grid_count in range(1, 6):
			# Compute Gauss–Legendre quadrature points and weights
			xGL, wGL = np.polynomial.legendre.leggauss(itheta_grid_count)

			# Print the Gauss–Legendre quadrature results in a single row
			print( colored( f"Gauss–Legendre quadrature for {itheta_grid_count} points:", HEADER_COLOR, attrs=[ 'bold', 'underline']))
			print( colored( f"  Quadrature Points (xGL): ", LABEL_COLOR) + colored( f"{xGL} ", VALUE_COLOR) + colored( f"| Corresponding Weights (wGL): ", LABEL_COLOR) + colored( f"{wGL}\n", VALUE_COLOR))

		# Print separator line
		print(colored("*" * 80, SEPARATOR_COLOR))

		# Flush output to ensure it's printed
		sys.stdout.flush()

	# Computation of Gaussian quadrature points and weights
	xGL, wGL = np.polynomial.legendre.leggauss(theta_grid_count)
	phixiGridPts = np.linspace(0, 2 * np.pi, phi_grid_count, endpoint=False)
	dphixi = 2. * np.pi / phi_grid_count

	return xGL, wGL, phixiGridPts, dphixi

def get_number_of_basis_functions_by_spin_states(max_angular_momentum_quantum_number, spin_state):
	"""
	Computes and displays the number of real spherical harmonic basis functions
	for a linear rotor system, categorized by nuclear spin isomer type.

	Parameters
	----------
	max_angular_momentum_quantum_number : int
		Maximum angular momentum quantum number (ℓ_max).
	spin_state : str
		Spin isomer type: "spinless", "para", or "ortho".

	Returns
	-------
	dict
		Dictionary with keys:
		- "JM" : Total number of |J,M> functions
		- "JeM": Number of even-J basis functions
		- "JoM": Number of odd-J basis functions
		- "JM_spin_specific": Number for selected spin state
	"""
	spin_state = spin_state.lower()

	if max_angular_momentum_quantum_number < 0:
		raise ValueError("max_angular_momentum_quantum_number must be non-negative.")

	# Total number of basis functions: sum over (2J + 1) from J = 0 to J_max
	JM = (max_angular_momentum_quantum_number + 1) ** 2

	# Even and odd J contributions
	if max_angular_momentum_quantum_number % 2 == 0:
		JeM = (JM + max_angular_momentum_quantum_number + 1) // 2
		JoM = JM - JeM
	else:
		JoM = (JM + max_angular_momentum_quantum_number + 1) // 2
		JeM = JM - JoM

	# Assign basis count based on isomer type
	if spin_state == "spinless":
		njm = JM
	elif spin_state == "para":
		njm = JeM
	elif spin_state == "ortho":
		njm = JoM
	else:
		raise ValueError("Invalid spin_state. Choose from 'spinless', 'para', or 'ortho'.")

	# Display summary
	print(colored("\nNumber of basis functions", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Total |J,M⟩ basis functions:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(JM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Even J basis functions (JeM):".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(JeM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Odd J basis functions (JoM):".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(JoM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"Basis functions for {spin_state} isomer:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(njm).ljust(VALUE_WIDTH), VALUE_COLOR))

	return {
		"JM": JM,
		"JeM": JeM,
		"JoM": JoM,
		"JM_spin_specific": njm
	}

def check_normalization_condition_linear_rotor(log_file, basis_description_text, basis_function_matrix_data, normalization_matrix_data, total_number_of_basis_functions, all_quantum_numbers, deviation_tolerance_value, file_write_mode="new"):
	"""
	Checks whether the normalization condition <JM|J'M'> ≈ δ_JJ'MM' holds.
	
	- Logs shape information of the basis function matrix and normalization matrix.
	- Uses a numerical check to verify if the normalization matrix is close to the identity matrix.
	- Identifies and logs only significant deviations from the expected values.
	- Computes eigenvalues for numerical stability analysis.
	
	Arguments:
		log_file (str): The path to save the normalization check log file.
		basis_description_text (str): A textual description of the basis used.
		basis_function_matrix_data (numpy.ndarray): The numerical matrix representing the basis functions.
		normalization_matrix_data (numpy.ndarray): The numerical matrix <JM|J'M'> representing normalization.
		total_number_of_basis_functions (int): The total number of basis functions used in the system.
		all_quantum_numbers (numpy.ndarray): The array containing quantum number pairs for each basis function.
		deviation_tolerance_value (float): The threshold value for detecting deviations from the expected identity matrix.
		file_write_mode (str): Either "new" to overwrite the file or "append" to add data to an existing file.
	"""
	# Determine whether to overwrite or append to the output file
	file_open_mode = "w" if file_write_mode == "new" else "a"

	# Compute the deviation of the normalization matrix from the identity matrix
	identity_matrix_reference = np.eye(total_number_of_basis_functions)
	deviation_from_identity_matrix = np.abs(normalization_matrix_data - identity_matrix_reference)
	maximum_deviation_value = np.max(deviation_from_identity_matrix)

	with open(log_file, file_open_mode) as output_file:
		# Write section header
		output_file.write("=" * 80 + "\n")
		output_file.write(f"Normalization Condition Check for {basis_description_text} \n")
		output_file.write("=" * 80 + "\n")

		# Write matrix dimensions
		output_file.write(f"Shape of basis function matrix: {basis_function_matrix_data.shape}\n")
		output_file.write(f"Shape of normalization matrix: {normalization_matrix_data.shape}\n\n")

		# Check if the normalization matrix is sufficiently close to the identity matrix
		if np.allclose(normalization_matrix_data, identity_matrix_reference, atol=deviation_tolerance_value):
			output_file.write("✔ Normalization check passed: The normalization matrix is close to the identity matrix.\n")
			output_file.write(f"Maximum deviation observed: {maximum_deviation_value:.3e}\n\n")
			return maximum_deviation_value
		else:
			output_file.write("Warning: Significant deviations detected in the normalization matrix.\n\n")

		output_file.write(f"Maximum deviation from the identity matrix: {maximum_deviation_value:.3e}\n\n")

		# Identify and log only significant deviations
		positions_of_significant_deviations = np.argwhere(deviation_from_identity_matrix > deviation_tolerance_value)
		if len(positions_of_significant_deviations) > 0:
			output_file.write("Significant Deviations Observed (Real Part, Imaginary Part):\n")
			for row_position, column_position in positions_of_significant_deviations:
				output_file.write(f"  Position ({row_position}, {column_position}): "
								  f"Real={np.real(normalization_matrix_data[row_position, column_position]):.5f}, "
								  f"Imaginary={np.imag(normalization_matrix_data[row_position, column_position]):.5f}\n")
			output_file.write("\n")

		# Compute eigenvalues of the normalization matrix for stability analysis
		eigenvalues_of_normalization_matrix = np.linalg.eigvalsh(normalization_matrix_data)
		deviation_of_eigenvalues_from_unity = np.abs(eigenvalues_of_normalization_matrix - 1)
		maximum_eigenvalue_deviation_value = np.max(deviation_of_eigenvalues_from_unity)
		mean_eigenvalue_deviation_value = np.mean(deviation_of_eigenvalues_from_unity)

		output_file.write("  Eigenvalue Analysis of Normalization Matrix:\n")
		output_file.write("  Computed Eigenvalues: " + " ".join([f"{computed_eigenvalue:.5f}" for computed_eigenvalue in eigenvalues_of_normalization_matrix]) + "\n")
		output_file.write(f"  Maximum Eigenvalue Deviation Observed: {maximum_eigenvalue_deviation_value:.3e}\n")
		output_file.write(f"  Mean Eigenvalue Deviation Observed: {mean_eigenvalue_deviation_value:.3e}\n")

		if maximum_eigenvalue_deviation_value > deviation_tolerance_value:
			output_file.write("[WARNING] Some eigenvalues deviate significantly from the expected value of 1!\n")

		output_file.write("=" * 80 + "\n")

	return maximum_deviation_value

def plot_heatmap(normalization_matrix_data, title):
	"""
	Visualizes the normalization matrix using a heatmap.
	
	Parameters:
		normalization_matrix_data (ndarray): The normalization matrix.
		title (str): Custom title for the heatmap.
	"""
	if normalization_matrix_data is None or normalization_matrix_data.size == 0:
		print("Error: The normalization matrix is empty or None!")
		return

	nrows, ncols = normalization_matrix_data.shape
	if nrows != ncols:
		print("Error: The normalization matrix must be square.")
		return

	# Set figure size
	plt.figure(figsize=(8, 6))

	# Choose whether to annotate based on size
	enable_annot = nrows <= 10

	sns.heatmap(
		np.abs(normalization_matrix_data),
		annot=enable_annot,
		cmap="viridis",
		linewidths=0.01,
		cbar=True
	)

	plt.title(f"{title}", fontsize=12, fontweight='bold')
	plt.xlabel("Basis Index", fontsize=10)
	plt.ylabel("Basis Index", fontsize=10)
	plt.tight_layout()
	plt.show()

def check_unitarity(file_name, spin_state, umat, small=1e-10, mode="new"):
	"""
	Checks whether the transformation matrix U satisfies the unitarity condition U U† = I.

	Parameters:
		file_name (str): Output file name for writing results.
		spin_state (str): Type of basis function.
		umat (ndarray): Transformation matrix.
		small (float): Numerical threshold for condition check.

	Returns:
		bool: True if U is unitary, False otherwise.
	"""
	# Determine file mode: "w" for first call, "a" for subsequent calls
	file_mode = "w" if mode == "new" else "a"

	# Compute U U†
	umat_unitarity = np.einsum('ia,ja->ij', umat, np.conjugate(umat))
	
	# Identity matrix for comparison
	identity_matrix = np.eye(umat.shape[0])

	# Compute deviation from identity
	deviation = np.linalg.norm(umat_unitarity - identity_matrix)

	with open(file_name, file_mode) as f:
		f.write(f"Checking Unitarity Condition for {spin_state} Transformation Matrix U\n")
		f.write("="*80 + "\n")
		f.write(f"Shape of {spin_state} U matrix: {umat.shape}\n")
		f.write(f"Max deviation from identity: {deviation}\n")

		if deviation < small:
			f.write("[INFO] The matrix U satisfies the unitarity condition.\n")
		else:
			f.write("[WARNING] The matrix U does NOT satisfy the unitarity condition.\n")

	return deviation < small  # Returns True if unitarity holds, False otherwise

def compute_rotational_kinetic_energy_loop(umat, all_quantum_numbers, B_const_K):
	"""
	Computes the rotational kinetic energy operator T_rot using a loop.

	Parameters:
	umat : numpy.ndarray
		Unitary normalization matrix (n_basis, n_basis).
	all_quantum_numbers : numpy.ndarray
		Array of quantum numbers where column 0 contains J values.
	B_const_K : float
		Rotational constant.

	Returns:
	numpy.ndarray
		Rotational kinetic energy operator matrix T_rot.
	"""
	n_basis = umat.shape[0]
	T_rot = np.zeros((n_basis, n_basis), dtype=complex)  # Ensure the matrix can hold complex values

	# Compute rotational energy levels B * J(J+1)
	rotational_energies = B_const_K * all_quantum_numbers[:, 0] * (all_quantum_numbers[:, 0] + 1)

	for jm in range(n_basis):
		for jmp in range(n_basis):
			sum_value = 0.0
			for s in range(n_basis):
				sum_value += umat[s, jm].conj() * umat[s, jmp] * rotational_energies[s]
			T_rot[jm, jmp] = sum_value

	#return np.real(T_rot)
	return T_rot

def compute_rotational_kinetic_energy_matrix(umat, all_quantum_numbers, B_const_K):
	"""
	Computes the rotational kinetic energy operator T_rot using efficient matrix operations.

	Parameters:
	umat : numpy.ndarray
		Unitary normalization matrix (n_basis, n_basis).
	all_quantum_numbers : numpy.ndarray
		Array of quantum numbers where column 0 contains J values.
	B_const_K : float
		Rotational constant.

	Returns:
	numpy.ndarray
		Rotational kinetic energy operator matrix T_rot.
	"""
	n_basis = umat.shape[0]

	# Validate Inputs
	if umat.shape[0] != umat.shape[1]:
		raise ValueError("Unitary normalization matrix U must be square.")
	if len(all_quantum_numbers) != n_basis:
		raise ValueError("Length of quantum numbers must match the dimensions of U.")

	# Compute rotational energy levels B * J(J+1)
	J_values = all_quantum_numbers[:, 0]
	rotational_energies = B_const_K * J_values * (J_values + 1)

	# Create a diagonal matrix from rotational energies
	E_diag = np.diag(rotational_energies)

	# Compute T_rot using matrix multiplication
	T_rot = umat.conj().T @ E_diag @ umat

	#return np.real(T_rot)  # Return the real part of the resulting matrix
	return T_rot  # Return the real part of the resulting matrix

def compute_rotational_kinetic_energy_einsum(umat, all_quantum_numbers, B_const_K, debug=False):
	"""
	Computes the rotational kinetic energy operator T_rot using efficient Einstein summation.

	Parameters:
	umat : numpy.ndarray
		Unitary normalization matrix (n_basis, n_basis).
	all_quantum_numbers : numpy.ndarray
		Array of quantum numbers where column 0 contains J values.
	B_const_K : float
		Rotational constant.
	debug : bool, optional
		If True, runs a debug check to verify that U^dagger * U = I (unitary property).

	Returns:
	numpy.ndarray
		Rotational kinetic energy operator matrix T_rot.
	"""
	# Number of basis functions (n_basis)
	n_basis = umat.shape[0]

	# Validate Inputs
	if umat.shape[0] != umat.shape[1]:
		raise ValueError("Unitary normalization matrix U must be square.")
	if len(all_quantum_numbers) != n_basis:
		raise ValueError("Length of quantum numbers must match the dimensions of U.")

	# Extract J values (rotational quantum numbers)
	J_values = all_quantum_numbers[:, 0]

	# Compute the rotational energy levels B * J(J+1)
	rotational_energies = B_const_K * J_values * (J_values + 1.0)

	# Create a diagonal matrix from the rotational energy levels
	E_diag = np.diag(rotational_energies)

	# Debugging check for unitary matrix property: U^dagger * U = I
	if debug:
		identity_matrix = np.eye(n_basis)
		T_rot = np.einsum('ji, jk, kl -> il', umat.conj(), identity_matrix, umat)
	else:
		# Compute T_rot using Einstein summation notation for efficient matrix multiplication
		T_rot = np.einsum('ji, jk, kl -> il', umat.conj(), E_diag, umat)

	return T_rot  # Return the real part of the resulting matrix

def check_hermiticity(H, matrix_name="H", description="", tol=1e-10, debug=True, visualize=False):
	"""
	Checks if a given matrix H is Hermitian and identifies discrepancies.

	Parameters:
	- H (np.ndarray): Input complex matrix.
	- matrix_name (str): Name of the matrix for debugging/visualization.
	- tol (float): Tolerance for detecting non-Hermitian elements.
	- debug (bool): If True, prints debugging information.
	- visualize (bool): If True, generates heatmap plots.

	Returns:
	- bool: True if H is Hermitian, False otherwise.
	- list: List of (row, col, discrepancy value) for non-Hermitian elements.
	"""

	if not isinstance(H, np.ndarray):
		raise TypeError("Input matrix H must be a NumPy array.")

	if H.size == 0:
		raise ValueError("Matrix H cannot be empty.")

	if H.shape[0] != H.shape[1]:
		raise ValueError("Matrix H must be square.")

	H_dagger = H.conj().T  # Compute Hermitian conjugate (H†)
	diff = np.abs(H - H_dagger)  # Absolute difference |H - H†|

	# Handle case where diff is empty
	max_diff = np.max(diff) if diff.size > 0 else 0.0
	norm_diff = np.linalg.norm(diff) if diff.size > 0 else 0.0
	mean_diff = np.mean(diff) if diff.size > 0 else 0.0

	# Find discrepancies
	discrepancy_indices = np.argwhere(diff > tol)
	discrepancies = [(int(i), int(j), float(diff[i, j])) for i, j in discrepancy_indices]

	is_hermitian = len(discrepancies) == 0  # True if no discrepancies exist

	if debug:
		print("\n**")
		print(colored(f"Hermiticity Check: {description}", HEADER_COLOR, attrs=['bold', 'underline']))
		print(f"[INFO] Matrix shape	 : {H.shape}")
		print(f"[INFO] Max deviation	: {max_diff:.2e}")
		print(f"[INFO] Frobenius norm   : {norm_diff:.2e}")
		print(f"[INFO] Mean deviation   : {mean_diff:.2e}")

		if discrepancies:
			print(f"\n[ERROR] {len(discrepancies)} discrepancies found (threshold = {tol}):")
			print("   Index (Row, Col)  | Deviation |")
			print("   --------------------------------")
			for i, j, value in discrepancies[:10]:  # Show first 10 discrepancies
				print(f"   ({i:3d}, {j:3d})   | {value:.2e}")
			if len(discrepancies) > 10:
				print("   ... (truncated)")

		else:
			print("[INFO] No discrepancies found. Matrix is Hermitian.")

	nrows, ncols = H.shape
	if nrows != ncols:
		print("Error: The normalization matrix must be square.")
		return

	if visualize:
		fig, axes = plt.subplots(1, 2, figsize=(18, 5))

		# Choose whether to annotate based on size
		enable_annot = nrows <= 10

		#sns.heatmap(H.real, cmap="coolwarm", annot=False, ax=axes[0])
		sns.heatmap(H.real, cmap="viridis", linewidths=0.1, annot=enable_annot, ax=axes[0])
		axes[0].set_title(f"Original Matrix (Re[{matrix_name}])")

		#sns.heatmap(H_dagger.real, cmap="coolwarm", annot=False, ax=axes[1])
		sns.heatmap(H_dagger.real, cmap="viridis", linewidths=0.1, annot=enable_annot, ax=axes[1])
		axes[1].set_title(f"Hermitian Conjugate (Re[{matrix_name}†])")

		plt.tight_layout()
		plt.show()

		fig, ax = plt.subplots(figsize=(12, 7))
		#sns.heatmap(diff, cmap="viridis", annot=True, fmt=".2e", ax=ax)
		sns.heatmap(diff, cmap="viridis", linewidths=0.1, annot=enable_annot, fmt=".2e", ax=ax)
		ax.set_title(f"Difference |{matrix_name} - {matrix_name}†| (Max: {max_diff:.2e})")

		plt.tight_layout()
		plt.show()

	return is_hermitian, discrepancies

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
	print(f"\n{'J':<5}{'Energy (Kelvin)':>15}")
	print("=" * 20)
	for J, E in energies.items():
		print(f"{J:<5}{E:>15.2f}")
	
	return energies

def plot_rotational_levels(energies):
	""" 
	Plots the rotational energy levels of a rigid rotor.
		
	Parameters:
	- energies (dict): Dictionary with J values as keys and energy values in Kelvin.
	"""
	J_values = list(energies.keys())
	energy_values = list(energies.values())

	plt.figure(figsize=(10, 6))
	plt.vlines(J_values, 0, energy_values, color='royalblue', linewidth=2)
	plt.scatter(J_values, energy_values, color='crimson', s=80, zorder=3, label="Energy Levels")

	# Annotate energy values above each level
	for J, E in energies.items():
		plt.text(J, E + max(energy_values) * 0.03, f"{E:.1f} K", ha='center', va='bottom', fontsize=10, color='black')

	plt.xticks(J_values)
	plt.yticks(fontsize=10)
	plt.xlabel("Rotational Quantum Number (J)", fontsize=12)
	plt.ylabel("Rotational Energy (K)", fontsize=12)
	plt.title("Rotational Energy Levels of a Rigid Rotor", fontsize=14, weight='bold')
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.tight_layout()
	plt.legend()
	plt.show()

if False:
	def plot_rotational_levels(energies):
		""" 
		Plots the rotational energy levels of a rigid rotor with enhanced aesthetics.
		
		Parameters:
		- energies (dict): Dictionary where keys are rotational quantum numbers (J) and
		  values are the corresponding energy levels in Kelvin.
		"""
		J_values = list(energies.keys())
		energy_values = list(energies.values())

		max_energy = max(energy_values)
		offset = max_energy * 0.04  # space for annotation

		plt.figure(figsize=(12, 6))
		
		# Plot vertical energy levels as steps
		for J in J_values:
			plt.hlines(energy_values[J], J - 0.3, J + 0.3, colors='teal', linewidth=3)
			plt.text(J, energy_values[J] + offset, f"{energy_values[J]:.2f} K", 
					 ha='center', va='bottom', fontsize=10, color='dimgray')

		# Formatting axes
		plt.xticks(J_values, fontsize=10)
		plt.yticks(fontsize=10)
		plt.xlabel("Rotational Quantum Number $J$", fontsize=12, weight='bold')
		plt.ylabel("Rotational Energy (K)", fontsize=12, weight='bold')
		plt.title("Rotational Energy Levels of a Rigid Rotor", fontsize=14, weight='bold')

		# Visual tweaks
		plt.grid(axis='y', linestyle='--', alpha=0.5)
		plt.xlim(min(J_values) - 1, max(J_values) + 1)
		plt.ylim(0, max_energy + 4 * offset)
		plt.tight_layout()
		plt.box(False)
		plt.show()

def extract_diagonal(matrix):
	"""
	Extracts the diagonal elements from a given matrix.

	Parameters:
	- matrix (np.ndarray): Input square or rectangular matrix.

	Returns:
	- np.ndarray: Array containing the diagonal elements.
	"""
	if not isinstance(matrix, np.ndarray):
		raise TypeError("Input must be a NumPy array.")
	
	return np.diagonal(matrix)  # Extract diagonal elements

def display_rotational_energies(diagonal_elements, all_quantum_numbers, B_const_K):
	"""
	Displays the extracted diagonal elements as rotational energy levels.

	Parameters:
	- diagonal_elements (np.ndarray): Extracted diagonal elements representing energy levels.
	- all_quantum_numbers (np.ndarray): Array of quantum numbers, where each row represents a state
											  and the first column contains the J values (rotational quantum numbers).
	- B_const_K (float): The rotational constant (cm⁻¹), used to compute rotational energy levels.

	Returns:
	None
	"""
	print("\nRotational Energy Levels")
	print("=" * 80)
	print(f"{'Quantum State (J)':^25} {'BJ(J+1) (Kelvin)':^25} {'<JM|T|JM> (Kelvin)':^25}")
	print("=" * 80)

	# Extracting J values from the quantum numbers data
	J_values = all_quantum_numbers[:, 0]

	# Compute the rotational energy levels B * J(J+1)
	for J, energy in zip(J_values, diagonal_elements):
		# Calculate the theoretical energy level based on the B constant
		theoretical_energy = B_const_K * J * (J + 1)
		
		
		# Display the results
		print(f"{int(J):>12} {theoretical_energy:>32.6f} {energy:>26.6f}")

	print("=" * 80)

def compute_potential_energy_einsum(basisfun_complex, umat, xGL, theta_grid_count, phi_grid_count, potential_strength, debug=False):
	"""
	Computes the potential energy operator in the rotational eigenbasis.

	Parameters:
	- basisfun_complex: Complex spherical harmonics basis functions.
	- umat: Unitary transformation matrix from (l, m) to (J, M) basis.
	- xGL: Gauss-Legendre quadrature nodes (cos(theta)).
	- theta_grid_count: Number of polar grid points.
	- phi_grid_count: Number of azimuthal grid points.
	- potential_strength: Scaling factor A in V(theta) = -A cos(theta).
	- debug: Flag to enable debugging (use constant potential if True).

	Returns:
	- V_rot: Potential energy operator matrix in the |J, M⟩ basis.
	"""
	
	# Compute the potential function on the grid (constant potential for now)
	if debug:
		# Using a constant potential (debug mode)
		pot_func_grid = np.ones(theta_grid_count * phi_grid_count)
	else:
		# Using the potential function based on xGL (cos(theta))
		pot_func_grid = np.repeat(xGL, phi_grid_count)

	# Scale the potential function by the potential strength
	pot_func_grid *= -potential_strength

	# Compute the potential matrix in the (l, m) basis using Einstein summation
	potential_matrix_complex_basis_function = np.einsum("gi,g,gj->ij", basisfun_complex.conj(), pot_func_grid, basisfun_complex)

	# Perform the transformation to the rotational basis using Einstein summation
	V_rot = np.einsum('ji,jk,kl->il', umat.conj(), potential_matrix_complex_basis_function, umat)

	return V_rot

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

def debug_eigenvalues_eigenvectors(H_rot, sorted_eigenvalues, sorted_eigenvectors):
	"""
	Validates the correctness of eigenvalues and eigenvectors computed from a Hamiltonian matrix.

	Parameters:
	- H_rot (ndarray): Original real symmetric or complex Hermitian Hamiltonian matrix.
	- sorted_eigenvalues (ndarray): 1D array of sorted eigenvalues.
	- sorted_eigenvectors (ndarray): 2D array of corresponding eigenvectors (each column is an eigenvector).

	Returns:
	- None: Performs checks and prints validation status.
	"""
	print("\n🔍 DEBUGGING EIGENVALUES AND EIGENVECTORS")

	# 1. Check Hermitian property of the Hamiltonian
	assert np.allclose(H_rot, H_rot.T.conj()), "[ERROR] H_rot is not Hermitian."
	print("[INFO] H_rot is Hermitian.")

	# 2. Verify eigenvalues are sorted in ascending order
	assert np.all(np.diff(sorted_eigenvalues) >= 0), "[ERROR] Eigenvalues are not sorted."
	print("[INFO] Eigenvalues are sorted.")

	# 3. Check orthonormality of eigenvectors
	identity = np.dot(sorted_eigenvectors.T.conj(), sorted_eigenvectors)
	assert np.allclose(identity, np.eye(identity.shape[0])), "[ERROR] Eigenvectors are not orthonormal."
	print("[INFO] Eigenvectors are orthonormal.")

	# 4. Validate the eigen-decomposition: H = VΛV†
	H_reconstructed = sorted_eigenvectors @ np.diag(sorted_eigenvalues) @ sorted_eigenvectors.T.conj()
	assert np.allclose(H_rot, H_reconstructed), "[ERROR] Eigen-decomposition reconstruction failed."
	print("[INFO] Hamiltonian reconstruction from eigenpairs is accurate.")

	# 5. Check for complex eigenvectors
	if np.iscomplexobj(sorted_eigenvectors):
		print("[WARNING] Eigenvectors contain complex numbers. Only real part may be stored in NetCDF.")

	print("[TARGET] All validations passed successfully.")

def save_all_quantum_data_to_netcdf(
	file_name,
	cm_inv_to_K,
	potential_strength_cm_inv,
	max_angular_momentum_quantum_number,
	theta_grid_count,
	phi_grid_count,
	B_const_cm_inv,
	spin_state,
	all_quantum_numbers,
	quantum_numbers_for_spin_state,
	sorted_eigenvalues,
	sorted_eigenvectors,
	dipole_moment_D=None,
	electric_field_kVcm=None
):
	"""
	Save quantum numbers, eigenvalues, and eigenvectors to a NetCDF file.

	Parameters:
	- file_name (str): Output NetCDF file name.
	- cm_inv_to_K (float): Conversion factor from cm-1 to Kelvin
	- potential_strength_cm_inv (float): Orienting potential strength in cm⁻¹.
	- max_angular_momentum_quantum_number (int): Truncation level ℓ_max.
	- theta_grid_count (int): Number of θ grid points.
	- phi_grid_count (int): Number of φ grid points.
	- B_const_cm_inv (float): Rotational constant in cm⁻¹.
	- spin_state (str): 'spinless', 'ortho', or 'para'.
	- all_quantum_numbers (ndarray): Full quantum number list.
	- quantum_numbers_for_spin_state (ndarray): Quantum numbers allowed for this spin state.
	- sorted_eigenvalues (ndarray): Eigenvalues (in cm⁻¹).
	- sorted_eigenvectors (ndarray): Complex eigenvectors.
	- dipole_moment_D (float, optional): Dipole moment in Debye.
	- electric_field_kVcm (float, optional): Electric field in kV/cm.
	"""

	sorted_eigenvalues = np.asarray(sorted_eigenvalues, dtype=np.float64)
	sorted_eigenvectors = np.asarray(sorted_eigenvectors, dtype=np.complex128)

	real_eigenvectors = np.real(sorted_eigenvectors)
	imag_eigenvectors = np.imag(sorted_eigenvectors)

	with Dataset(file_name, "w", format="NETCDF4") as ncfile:
		# Metadata and scalar parameters
		write_metadata(ncfile, spin_state)
		write_scalar_parameters(
			ncfile,
			cm_inv_to_K,
			potential_strength_cm_inv,
			max_angular_momentum_quantum_number,
			spin_state,
			theta_grid_count,
			phi_grid_count,
			B_const_cm_inv
		)

		# Dipole-field data if applicable
		if dipole_moment_D is not None:
			ncfile.dipole_moment_D = dipole_moment_D
		if electric_field_kVcm is not None:
			ncfile.electric_field_kVcm = electric_field_kVcm
		if dipole_moment_D is not None and electric_field_kVcm is not None:
			ncfile.muE_cm_inv = dipole_moment_D * electric_field_kVcm * 0.03065

		# Store quantum data
		write_quantum_numbers(ncfile, all_quantum_numbers, spin_state, quantum_numbers_for_spin_state)
		write_eigen_data(ncfile, sorted_eigenvalues, real_eigenvectors, imag_eigenvectors)

	# Output confirmation
	print("\n**")
	print(
		colored("NetCDF Output File:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		colored(f"{file_name}".ljust(VALUE_WIDTH), VALUE_COLOR)
	)

def write_metadata(ncfile, spin_state_name):
	"""
	Write metadata into the given NetCDF file for quantum rotor simulations.

	Parameters:
	- ncfile (netCDF4.Dataset): The open NetCDF file object.
	- spin_state_name (str): Type of spin isomer ('spinless', 'ortho', or 'para').
	"""
	# --- User & Host Information ---
	username = getpass.getuser()
	hostname = socket.getfqdn()

	try:
		ip_address = socket.gethostbyname(socket.gethostname())
	except socket.gaierror:
		ip_address = "unavailable"

	os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
	python_version = sys.version.replace('\n', ' ')

	# --- Metadata Block ---
	ncfile.title = "Quantum Rotational States and Eigenvalue Data"
	ncfile.description = (
		f"Eigenvalues and eigenfunctions computed via exact diagonalization "
		f"for a linear quantum rotor system under orienting potential. "
		f"Spin isomer: '{spin_state_name}'."
	)
	ncfile.source = "Simulation using real spherical harmonics basis and exact diagonalization"
	ncfile.institution = "National Institute of Technology Raipur"
	ncfile.history = f"Created on {datetime.now().isoformat()} by {username} on host '{hostname}'"
	ncfile.license = "This data is provided solely for academic and research use."
	ncfile.conventions = "CF-1.6"

	# --- Simulation & Runtime Metadata ---
	ncfile.spin_isomer = spin_state_name
	ncfile.creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	ncfile.creator_name = username
	ncfile.creator_host = hostname
	ncfile.creator_ip = ip_address
	ncfile.operating_system = os_info
	ncfile.python_version = python_version

def write_scalar_parameters(
	ncfile,
	cm_inv_to_K,
	potential_strength_cm_inv,
	max_angular_momentum_quantum_number,
	spin_state_name,
	theta_grid_count,
	phi_grid_count,
	B_const_cm_inv
):
	"""
	Write scalar simulation parameters to the NetCDF file with appropriate units.

	Parameters:
	- ncfile: Open NetCDF file handle.
	- cm_inv_to_K (float): Conversion factor from cm-1 to Kelvin
	- potential_strength_cm_inv (float): Orienting potential strength in cm⁻¹.
	- max_angular_momentum_quantum_number (int): Truncation level ℓ_max.
	- spin_state_name (str): Spin isomer type ('spinless', 'ortho', or 'para').
	- theta_grid_count (int): Number of grid points in θ.
	- phi_grid_count (int): Number of grid points in φ.
	- B_const_cm_inv (float): Rotational constant in cm⁻¹.
	"""
	# Create a dummy scalar dimension
	ncfile.createDimension('scalar', 1)

	# cm_inv_to_K
	var_conversion_factor = ncfile.createVariable("cm_inv_to_K", "f8", ("scalar",))
	var_conversion_factor.units = "Kelvin/cm⁻¹"
	var_conversion_factor[0] = cm_inv_to_K

	# Potential strength
	var_potential = ncfile.createVariable("potential_strength_cm_inv", "f8", ("scalar",))
	var_potential.units = "cm⁻¹"
	var_potential[0] = potential_strength_cm_inv

	# Max J
	var_max_J = ncfile.createVariable("max_angular_momentum_quantum_number", "i4", ("scalar",))
	var_max_J.units = "dimensionless"
	var_max_J[0] = max_angular_momentum_quantum_number

	# Spin state
	var_spin = ncfile.createVariable("spin_state_name", str, ("scalar",))
	var_spin.units = "dimensionless"
	var_spin[0] = spin_state_name

	# Theta grid
	var_theta = ncfile.createVariable("theta_grid_count", "i4", ("scalar",))
	var_theta.units = "dimensionless"
	var_theta[0] = theta_grid_count

	# Phi grid
	var_phi = ncfile.createVariable("phi_grid_count", "i4", ("scalar",))
	var_phi.units = "dimensionless"
	var_phi[0] = phi_grid_count

	# Rotational constant
	var_B = ncfile.createVariable("rotational_constant_cm_inv", "f8", ("scalar",))
	var_B.units = "cm⁻¹"
	var_B[0] = B_const_cm_inv

def write_quantum_numbers(ncfile, all_quantum_numbers, spin_state_name, quantum_numbers_for_spin_state):
	"""
	Write the quantum numbers (J, M, etc.) to the NetCDF file for both full and spin-specific basis.

	Parameters:
	- ncfile: NetCDF file handle.
	- all_quantum_numbers (ndarray): All basis quantum numbers (e.g., J, M).
	- spin_state_name (str): Spin isomer type ('spinless', 'ortho', or 'para').
	- quantum_numbers_for_spin_state (ndarray): Subset of quantum numbers allowed for this spin state.
	"""
	all_quantum_numbers = np.array(all_quantum_numbers, dtype=np.int32)
	quantum_numbers_for_spin_state = np.array(quantum_numbers_for_spin_state, dtype=np.int32)

	# Create dimensions
	ncfile.createDimension("all_entries", all_quantum_numbers.shape[0])
	ncfile.createDimension("components", all_quantum_numbers.shape[1])
	ncfile.createDimension("spin_count", quantum_numbers_for_spin_state.shape[0])

	# Store all quantum numbers
	var_all_qn = ncfile.createVariable("all_quantum_numbers", "i4", ("all_entries", "components"))
	var_all_qn[:, :] = all_quantum_numbers
	var_all_qn.long_name = "All basis quantum numbers (e.g., J, M)"

	# Store quantum numbers specific to spin state
	var_spin_qn = ncfile.createVariable(f"{spin_state_name}_quantum_numbers", "i4", ("spin_count", "components"))
	var_spin_qn[:, :] = quantum_numbers_for_spin_state
	var_spin_qn.long_name = f"Quantum numbers for spin isomer '{spin_state_name}'"

	# Pretty print (only if J is reasonably small for visual clarity)
	max_J = np.max(all_quantum_numbers[:, 0])
	if max_J <= 4:
		print(colored("\nAll Quantum Numbers (J, M)", HEADER_COLOR, attrs=['bold', 'underline']))
		print(pd.DataFrame(all_quantum_numbers, columns=["J", "M"]))

		print(colored(f"\nSpin-Specific Quantum Numbers ({spin_state_name})", HEADER_COLOR, attrs=['bold', 'underline']))
		print(pd.DataFrame(quantum_numbers_for_spin_state, columns=["J", "M"]))

def write_eigen_data(ncfile, eigenvalues, real_eigenvectors, imag_eigenvectors):
	"""
	Write eigenvalues and eigenvectors to the NetCDF file.

	Parameters:
	- ncfile: NetCDF file handle.
	- eigenvalues (ndarray): Real-valued eigenvalues (in Kelvin).
	- real_eigenvectors (ndarray): Real parts of eigenvectors.
	- imag_eigenvectors (ndarray): Imaginary parts of eigenvectors.
	"""
	eigenvalues = np.array(eigenvalues, dtype=np.float64)
	real_eigenvectors = np.array(real_eigenvectors, dtype=np.float64)
	imag_eigenvectors = np.array(imag_eigenvectors, dtype=np.float64)

	state_count = eigenvalues.shape[0]
	vector_dim = real_eigenvectors.shape[1]

	# Create NetCDF dimensions
	ncfile.createDimension("state_count", state_count)
	ncfile.createDimension("vector_dim", vector_dim)

	# Store eigenvalues
	var_eigenvalues = ncfile.createVariable("eigenvalues", "f8", ("state_count",))
	var_eigenvalues[:] = eigenvalues
	var_eigenvalues.units = "Kelvin"
	var_eigenvalues.long_name = "Eigenvalues of the Hamiltonian in energy units"

	# Store eigenvector components
	var_eigvec_real = ncfile.createVariable("eigenvectors_real", "f8", ("state_count", "vector_dim"))
	var_eigvec_real[:, :] = real_eigenvectors
	var_eigvec_real.long_name = "Real part of Hamiltonian eigenvectors"

	var_eigvec_imag = ncfile.createVariable("eigenvectors_imag", "f8", ("state_count", "vector_dim"))
	var_eigvec_imag[:, :] = imag_eigenvectors
	var_eigvec_imag.long_name = "Imaginary part of Hamiltonian eigenvectors"

	# Display output for small systems only
	if state_count <= 50:
		print(colored("\nEigenvalues (in Kelvin)", HEADER_COLOR, attrs=['bold', 'underline']))
		print(pd.DataFrame(eigenvalues, columns=["Energy"]))

def main():
	# --- Parse command-line arguments ---
	args = parse_arguments()

	# --- Dry run mode: Show parameters and exit ---
	if args.dry_run:
		print(colored("=" * (LABEL_WIDTH + VALUE_WIDTH), SEPARATOR_COLOR))
		print(colored("Dry Run Summary".center(LABEL_WIDTH + VALUE_WIDTH), HEADER_COLOR))
		print(colored("=" * (LABEL_WIDTH + VALUE_WIDTH), SEPARATOR_COLOR))

		print(colored("ℓ_max".ljust(LABEL_WIDTH), LABEL_COLOR) + 
			  colored(str(args.max_angular_momentum_quantum_number).ljust(VALUE_WIDTH), VALUE_COLOR))
		
		print(colored("Spin".ljust(LABEL_WIDTH), LABEL_COLOR) + 
			  colored(args.spin.ljust(VALUE_WIDTH), VALUE_COLOR))
		
		print(colored("Dipole moment".ljust(LABEL_WIDTH), LABEL_COLOR) + 
			  colored(f"{args.dipole_moment or 'N/A'} D".ljust(VALUE_WIDTH), VALUE_COLOR))
		
		print(colored("Electric field".ljust(LABEL_WIDTH), LABEL_COLOR) + 
			  colored(f"{args.electric_field or 'N/A'} kV/cm".ljust(VALUE_WIDTH), VALUE_COLOR))
		
		print(colored("V(θ) strength".ljust(LABEL_WIDTH), LABEL_COLOR) + 
			  colored(f"{args.potential_strength or 'N/A'} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))
		
		print(colored("Output directory".ljust(LABEL_WIDTH), LABEL_COLOR) + 
			  colored(args.output_dir.ljust(VALUE_WIDTH), VALUE_COLOR))

		print(colored("=" * (LABEL_WIDTH + VALUE_WIDTH), SEPARATOR_COLOR))
		sys.exit(0)

	# --- Create output directory ---
	os.makedirs(args.output_dir, exist_ok=True)

	# --- Determine potential strength ---
	if args.potential_strength is not None:
		potential_strength_cm_inv = args.potential_strength
	elif args.dipole_moment_D is not None and args.electric_field_kVcm is not None:
		potential_strength_cm_inv = args.dipole_moment_D * args.electric_field_kVcm * 0.03065
		print(f"Computed potential strength (μE) = {potential_strength_cm_inv:.4f} cm⁻¹")
	else:
		raise ValueError(
			"Missing potential parameters: provide either --potential-strength "
			"or both --dipole-moment-D and --electric-field-kVcm."
		)

	max_angular_momentum_quantum_number = args.max_angular_momentum_quantum_number
	spin_state					= args.spin

	# No. of grid points along theta and phi
	theta_grid_count			= int(2 * max_angular_momentum_quantum_number + 5)
	phi_grid_count				= int(2 * theta_grid_count + 5)

	# Tolerance limit for a harmitian matrix
	deviation_tolerance_value   = 10e-12

	# print the normalization
	display_legendre_quadrature = False
	compute_rigid_rotor_energy  = False
	orthonormality_check		= False
	unitarity_check				= False
	hermiticity_check			= False
	pot_write					= False
	#
	display_data				= False

	# Display input parameters
	show_simulation_details(
	potential_strength_cm_inv=potential_strength_cm_inv,   # float, in cm⁻¹
	max_angular_momentum_quantum_number=args.max_angular_momentum_quantum_number,
	spin_state=args.spin,
	theta_grid_count=theta_grid_count,					 # e.g., 100
	phi_grid_count=phi_grid_count,						 # e.g., 120
	dipole_moment_D=args.dipole_moment,					 # float or None
	electric_field_kVcm=args.electric_field,			 # float or None
	computed_muE_cm_inv=potential_strength_cm_inv if args.potential_strength is None else None
	)

	# Spectroscopic constant (B) in cm⁻¹ taken from NIST data
	#B_const_cm_inv = 20.95373 # HF
	B_const_cm_inv = 10.44 # https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=josa&uri=josa-52-1-1&html=true

	# Retrieve the inverse meter-Kelvin relationship from physical constants
	m_inv_to_K, unit, uncertainty = const.physical_constants["inverse meter-kelvin relationship"]

	# Convert from inverse meters (m⁻¹) to inverse centimeters (cm⁻¹) using the relation: 1 m⁻¹ = 100 cm⁻¹
	cm_inv_to_K = m_inv_to_K / const.centi  

	# Compute the corresponding value in Kelvin
	B_const_K = B_const_cm_inv * cm_inv_to_K  
	potential_strength_K = potential_strength_cm_inv * cm_inv_to_K

	# Unit Conversion
	# Display results with clear labels and scientific precision
	print(colored("\nUnit Conversion", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored(f"Inverse meter-Kelvin relationship:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{m_inv_to_K:.8f} {unit} (± {uncertainty:.6e})".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"Conversion factor from cm⁻¹ to Kelvin:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{cm_inv_to_K:.6f} K/cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"Rotational constant in Kelvin:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{B_const_K:.6f} K".ljust(VALUE_WIDTH), VALUE_COLOR))

	if compute_rigid_rotor_energy:
		energies = rotational_energy_levels(B_const_K, 10)
		plot_rotational_levels(energies)

	# Gauss-Quadrature points
	xGL, wGL, phixiGridPts, dphixi = compute_legendre_quadrature(theta_grid_count, phi_grid_count, display_legendre_quadrature)

	base_file_name = generate_filename(spin_state, max_angular_momentum_quantum_number, potential_strength_K, theta_grid_count, phi_grid_count, args.dipole_moment, args.electric_field, prefix="_")
	
	# Ensure output directory exists
	os.makedirs(args.output_dir, exist_ok=True)
	# Output file name
	log_file = os.path.join(args.output_dir, f"validation_fundamental_QM_properties" + base_file_name + ".log")

	print(colored("\nValidation Log Filename", HEADER_COLOR, attrs=['bold', 'underline']))
	print( colored("Orthonormality Check Log:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(log_file.ljust(VALUE_WIDTH), VALUE_COLOR))

	# All quantum numbers: (J, M)
	all_quantum_numbers = bfunc.generate_linear_rotor_quantum_numbers(max_angular_momentum_quantum_number, "spinless")
	# Spin-state-specific quantum numbers
	quantum_numbers_for_spin_state = bfunc.generate_linear_rotor_quantum_numbers(max_angular_momentum_quantum_number, spin_state)

	basis_functions_info = get_number_of_basis_functions_by_spin_states(max_angular_momentum_quantum_number, spin_state)
	total_number_of_states = basis_functions_info["JM"]
	total_number_of_spin_states = basis_functions_info["JM_spin_specific"]
	
	# Real spherical harmonics basis <cos(θ), φ | JM> as a 2D matrix 'basisfun_real' with shape (theta_grid_count * phi_grid_count, n_basis), 
	# where each column corresponds to a unique (J, M) quantum number pair and rows map to grid points across θ and φ angles.
	n_basis_real = total_number_of_states
	basisfun_real = bfunc.spherical_harmonicsReal(n_basis_real, theta_grid_count, phi_grid_count, all_quantum_numbers, xGL, wGL, phixiGridPts, dphixi)

	# Section Header
	print(colored("\nBasis Function Shapes", HEADER_COLOR, attrs=['bold', 'underline']))

	# Real basis functions
	print(colored("Real (basisfun_real):".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(f"{basisfun_real.shape}".ljust(VALUE_WIDTH), VALUE_COLOR))


	if (orthonormality_check):
		# Compute the overlap (normalization) matrix to check if the basis functions are orthonormal.  
		# The resulting real_basis_normalization_matrix is of size (n_basis, n_basis), where n_basis is the number of basis functions.  
		# If the basis functions are perfectly normalized and orthogonal, real_basis_normalization_matrix should be close to the identity matrix.  
		real_basis_normalization_matrix = np.einsum('ij,ik->jk', basisfun_real, basisfun_real)  # (n_points, n_basis) x (n_points, n_basis) → (n_basis, n_basis)
		#real_basis_normalization_matrix = np.tensordot(basisfun_real, np.conjugate(basisfun_real), axes=([0], [0]))
		#df = pd.DataFrame(real_basis_normalization_matrix)
		#print(df)
		
		#
		print(colored("shape of ", INFO_COLOR) + colored("real_basis_normalization_matrix: ".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{real_basis_normalization_matrix.shape}".ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")

		basis_description_text = "Real Spherical Harmonics Basis |JM> for a linear rigid rotor"
		check_normalization_condition_linear_rotor(
			log_file,
			basis_description_text,
			basisfun_real,
			real_basis_normalization_matrix,
			n_basis_real,
			all_quantum_numbers,
			deviation_tolerance_value,
			file_write_mode="new"
		)
		title = f"Heatmap of normalization matrix \n Real basis"
		plot_heatmap(real_basis_normalization_matrix, title)

		is_Hermitian, max_diff = check_hermiticity(real_basis_normalization_matrix, "S", "Real Normalization Matrix", tol=1e-10, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")


	n_basis_complex = total_number_of_states
	# Construction of complex basis functions 
	basisfun_complex = bfunc.spherical_harmonicsComp(n_basis_complex, theta_grid_count, phi_grid_count, all_quantum_numbers, xGL, wGL, phixiGridPts, dphixi)
	# Complex basis functions, if available
	if 'basisfun_complex' in locals() or 'basisfun_complex' in globals():
		print(colored("Complex (basisfun_complex):".ljust(LABEL_WIDTH), LABEL_COLOR) +
			  colored(f"{basisfun_complex.shape}".ljust(VALUE_WIDTH), VALUE_COLOR))

	if (orthonormality_check):
		# Orthonormality test for "complex basis"
		complex_basis_normalization_matrix = np.einsum('ij,ik->jk', np.conjugate(basisfun_complex), basisfun_complex)  # (n_points, n_basis) x (n_points, n_basis) → (n_basis, n_basis)
		basis_description_text = "Complex Spherical Harmonics Basis |JM> for a linear rigid rotor"
		check_normalization_condition_linear_rotor(
			log_file,
			basis_description_text,
			basisfun_complex,
			complex_basis_normalization_matrix,
			n_basis_complex,
			all_quantum_numbers,
			deviation_tolerance_value,
			file_write_mode="append"
		)
		title = f"Heatmap of normalization matrix \n Complex basis"
		plot_heatmap(complex_basis_normalization_matrix, title)

		is_Hermitian, max_diff = check_hermiticity(complex_basis_normalization_matrix, "S", "Complex Normalization Matrix", tol=1e-10, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

	#
	# Construction of Unitary Matrix 
	# umat = np.tensordot(np.conjugate(basisfun_complex), basisfun_real, axes=([0], [0]))
	umat = np.einsum('ij,ik->jk', np.conjugate(basisfun_complex), basisfun_real)
	#umat = basisfun_complex.conj().T @ basisfun_real

	if (unitarity_check):
		check_unitarity(log_file, spin_state, umat, mode="append")
		# Compute UU†
		umat_unitarity = np.einsum('ia,ja->ij', umat, np.conjugate(umat))
		#umat_unitarity = np.einsum('ia,ja->ij', umat, umat.conj())
		#umat_unitarity = np.einsum('ia,ib->ab', umat, umat.conj())
		#umat_unitarity = umat.conj().T @ umat
		title = f"Heatmap of UU† matrix for {spin_state} spin state"
		plot_heatmap(umat_unitarity, title)

		is_Hermitian, max_diff = check_hermiticity(umat_unitarity, "(UU†)", "Complex UU† Matrix", tol=1e-40, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

	# Call the function to compute the rotational kinetic energy operator
	#T_rot_einsum = compute_rotational_kinetic_energy_einsum(umat, all_quantum_numbers, B_const_K)
	T_rot_einsum = compute_rotational_kinetic_energy_einsum(umat, all_quantum_numbers, B_const_K, debug=False)
	if display_data:
		# Extract and display rotational energies
		diagonal_energies = extract_diagonal(T_rot_einsum.real)
		display_rotational_energies(diagonal_energies, all_quantum_numbers, B_const_K)

	if False:
		V_rot_1 = np.einsum('ia,ja->ij', umat, umat.conj())
		V_rot_2 = np.einsum('ia,ib->ab', umat, umat.conj())

		is_Hermitian, max_diff = check_hermiticity(V_rot, "V", tol=1e-10, debug=True, visualize=False)
	V_rot_einsum = compute_potential_energy_einsum(basisfun_complex, umat, xGL, theta_grid_count, phi_grid_count, potential_strength_K, debug=False)
	H_rot = T_rot_einsum + V_rot_einsum

	if hermiticity_check: 
		is_Hermitian, max_diff = check_hermiticity(T_rot_einsum, "T", "Rotational Kinetic Energy Matrix", tol=1e-10, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

		is_Hermitian, max_diff = check_hermiticity(V_rot_einsum, "V", "Potential Energy Matrix", tol=1e-10, debug=True, visualize=False)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

		is_Hermitian, max_diff = check_hermiticity(H_rot, "H", "Hamiltonian Matrix", tol=1e-10, debug=True, visualize=False)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

	# Compute eigenvalues and eigenvectors
	sorted_eigenvalues, sorted_eigenvectors = compute_sorted_eigenvalues_and_eigenvectors(H_rot)
	print(sorted_eigenvalues)
	for i, val in enumerate(sorted_eigenvalues):
		print(f"  Level {i}: {val:.6f}")

	# Debugging function call
	debug_eigenvalues_eigenvectors(H_rot, sorted_eigenvalues, sorted_eigenvectors)

	# Output file name
	file_name_netcdf = os.path.join(args.output_dir, f"quantum_data" + base_file_name + ".nc")

	# Prepare arguments
	kwargs = {
		"file_name": file_name_netcdf,
		"cm_inv_to_K": cm_inv_to_K,
		"potential_strength_cm_inv": potential_strength_cm_inv,
		"max_angular_momentum_quantum_number": max_angular_momentum_quantum_number,
		"theta_grid_count": theta_grid_count,
		"phi_grid_count": phi_grid_count,
		"B_const_cm_inv": B_const_cm_inv,
		"spin_state": spin_state,
		"all_quantum_numbers": all_quantum_numbers,
		"quantum_numbers_for_spin_state": quantum_numbers_for_spin_state,
		"sorted_eigenvalues": sorted_eigenvalues,
		"sorted_eigenvectors": sorted_eigenvectors,
	}

	# Conditionally add optional values
	if args.dipole_moment is not None and args.electric_field is not None:
		kwargs["dipole_moment_D"] = args.dipole_moment
		kwargs["electric_field_kVcm"] = args.electric_field

	# Call the function
	save_all_quantum_data_to_netcdf(**kwargs)
	
	print("\n\nHURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES")

if __name__ == "__main__":
	main()
