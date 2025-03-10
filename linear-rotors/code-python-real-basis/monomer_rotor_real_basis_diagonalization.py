# ************************************************************************************************#
#																								  #
# Computation of Eigenvalues and Eigenfunctions of a Linear Rotor Using Real Spherical Harmonics. #
#																								  #
# Developed by Dr. Tapas Sahoo																	  #
#																								  #
# ------------------------------------------------------------------------------------------------#
#																								  #
# Command for running the code:																	  #
#																								  #
# Example:																						  #
# python monomer_rotor_real_basis_diagonalization.py 10.0 2 spinless							  #
#																								 #
# ------------------------------------------------------------------------------------------------#
#																								 #
# Inputs:																						  #
# a) Potential potential_strength = potential_strength																  #
# b) Highest value of Angular quantum number = max_angular_momentum												  #
# c) Specification of spin isomer = spin_state													  #
#																								 #
# Outputs: Eigenvalues and eigenfunctions														  #
#																								 #
# ************************************************************************************************#

import argparse
import os
import sys
import getpass
import math
import numpy as np
from numpy.linalg import eigh
import scipy
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

# Imports basis functions of rotors (linear and nonlinear rotors)
import pkg_basis_func_rotors.basis_func_rotors as bfunc

# Define color schemes
HEADER_COLOR = 'cyan'
LABEL_COLOR = 'green'
VALUE_COLOR = 'magenta'
DEBUG_COLOR = 'red'
SEPARATOR_COLOR = 'yellow'
INFO_COLOR = 'blue'

# Define fixed widths for labels and values
LABEL_WIDTH = 35
VALUE_WIDTH = 45


def parse_arguments():
	"""Parse command-line arguments for potential potential_strength, max_angular_momentum, and spin isomer."""

	# Initialize parser for command-line arguments
	parser = argparse.ArgumentParser(
		prog="monomer_rotor_real_basis_diagonalization.py",
		description="Computation of Eigenvalues and Eigenfunctions of a Linear Rotor Using Real Spherical Harmonics.",
		epilog="Enjoy the program! :)")

	# Define the arguments with clear help messages and types
	parser.add_argument(
		"potential_strength",
		type=float,
		help="Interaction potential_strength of the potential in the form A*cos(theta). Enter a real number."
	)

	parser.add_argument(
		"max_angular_momentum",
		type=int,
		help="Truncated angular quantum number for the computation. Must be a non-negative integer."
	)

	parser.add_argument(
		"spin",
		type=str,
		choices=["para", "ortho", "spinless"],
		help="Specifies nuclear spin isomerism. Choose 'para', 'ortho', or 'spinless'."
	)

	# Parse the arguments
	return parser.parse_args()


def whoami():
	print('*' * 80)
	print("\nATTENTION: \n")
	print("%s/%s%s" % ("The function is \n" + sys._getframe(1).f_code.co_filename, sys._getframe(1).f_code.co_name, "\nand the line number is " + str(sys._getframe(1).f_lineno)))
	print("")
	print('*' * 80)
	exit()


def show_simulation_details(potential_potential_strength, max_angular_momentum, spin_state, theta_grid_count, phi_grid_count):
	"""Display the input parameters for the simulation process."""

	now = datetime.now()  # Current date and time
	date_time = now.strftime("%d/%m/%Y, %H:%M:%S")

	user_name = getpass.getuser()
	input_dir_path = os.getcwd()
	home_directory = os.path.expanduser("~")

	# Separator line
	print(colored("*" * 80, SEPARATOR_COLOR))

	# Date and Time Section
	print(colored("Date and Time:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(date_time.ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")

	# Debugging Information
	print(colored("Debug mode is enabled.", DEBUG_COLOR) + "\n")
	print("**")
	print(colored("File System Details:", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("User Name:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(user_name.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Home Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(home_directory.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Current Working Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(input_dir_path.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Package Location (bfunc):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(bfunc.__file__.ljust(VALUE_WIDTH), VALUE_COLOR))
	print("\n**")

	# Input Parameters Section
	print(colored("Simulation Parameters", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Potential potential_strength:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(str(potential_potential_strength).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Max Angular Momentum (J_max):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{max_angular_momentum}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Spin State:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{spin_state}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print("\n**")

	# Grid Information
	print(colored("Grid Information", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Theta Grid Count:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{theta_grid_count}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Phi Grid Count:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{phi_grid_count}".ljust(VALUE_WIDTH), VALUE_COLOR))

def generate_filename(
		spin_state: str,
		max_angular_momentum: int,
		potential_strength: float,
		theta_grid_count: int,
		phi_grid_count: int,
		prefix: Optional[str] = ""
) -> str:
	"""
	Generates a descriptive filename based on parameters for a linear rotor system.

	Parameters:
	- spin_state (str): The spin isomer type ("spinless", "para", or "ortho").
	- max_angular_momentum (int): Highest angular quantum number.
	- potential_strength (float): Field potential_strength in Kelvin.
	- theta_grid_count (int): Number of theta grids.
	- phi_grid_count (int): Number of phi grids.
	- prefix (str, optional): Directory path or prefix for the file name. Defaults to "".

	Returns:
	- str: Constructed file name.
	"""
	# Determine isomer and basis type based on spin isomer
	if spin_state == "spinless":
		isomer = "spinless"
		basis_type = "none"
	elif spin_state == "para":
		isomer = "para"
		basis_type = "even"
	elif spin_state == "ortho":
		isomer = "ortho"
		basis_type = "odd"
	else:
		raise ValueError(
			"Unknown spin isomer type: expected 'spinless', 'para', or 'ortho'.")

	# Construct the file name in a logical, readable format
	filename = (
		f"{prefix}_for_H2_{isomer}_isomer_max_angular_momentum{max_angular_momentum}_"
		f"potential_strength{potential_strength}K_"
		f"grids_theta{theta_grid_count}_phi{phi_grid_count}.txt"
	)

	return basis_type, filename


def compute_legendre_quadrature(theta_grid_count, phi_grid_count, io_write):
	"""
	Computes Gaussian quadrature points and weights for Legendre polynomials,
	along with phi grid points for a specified number of theta and phi points.

	Parameters:
	- theta_grid_count (int): Number of theta points for Legendre quadrature.
	- phi_grid_count (int): Number of phi points for uniform grid.
	- io_write (bool): If True, prints the quadrature points and weights.

	Returns:
	- xGL (np.ndarray): Gaussian quadrature points.
	- wGL (np.ndarray): Corresponding weights.
	- phixiGridPts (np.ndarray): Uniformly spaced phi grid points.
	- dphixi (float): Phi grid spacing.


	Gauss-Legendre quadrature points are given in https://en.wikipedia.org/wiki/Gaussian_quadrature
	"""

	# Optionally print the quadrature points and weights
	if io_write:
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


def get_number_of_basis_functions_by_spin_states(max_angular_momentum, spin_state):
	"""
	Gets and displays the number of basis functions for a linear rotor
	categorized by spin isomers (spinless, para, ortho).

	Parameters:
	- max_angular_momentum (int): The highest angular quantum number.
	- spin_state (str): The spin isomer type ("spinless", "para", or "ortho").

	Returns:
	- dict: A dictionary with JM, JeM, JoM, and njm values.
	"""
	# Calculate the total number of basis functions
	# JKM = "Sum[(2J+1),{J,0,max_angular_momentum}]" -- Derivation is given in
	# lecture-notes-on-exact-diagonalization.pdf or you can derive it on
	# ChatGPT
	JM = int((max_angular_momentum + 1)**2)

	# Determine the even (JeM) and odd (JoM) basis function counts
	if (max_angular_momentum % 2) == 0:
		JeM = int((JM + max_angular_momentum + 1) / 2)
		JoM = JM - JeM
	else:
		JoM = int((JM + max_angular_momentum + 1) / 2)
		JeM = JM - JoM

	# Assign njm based on the spin isomer
	if spin_state == "spinless":
		njm = JM
	elif spin_state == "para":
		njm = JeM
	elif spin_state == "ortho":
		njm = JoM
	else:
		raise ValueError( "Invalid spin isomer type. Choose from 'spinless', 'para', or 'ortho'.")

	# Optionally print the calculations if io_write is True
	print("\n**")
	print( colored( "Number of basis functions", HEADER_COLOR, attrs=[ 'bold', 'underline']))
	print( colored( "Total |JM> basis functions:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored( str(JM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print( colored( "Even J basis functions (JeM):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored( str(JeM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print( colored( "Odd J basis functions (JoM):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored( str(JoM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"Number of basis functions for {spin_state} isomer:".ljust( LABEL_WIDTH), LABEL_COLOR) + colored(str(njm).ljust(VALUE_WIDTH), VALUE_COLOR))

	return {
		"JM": JM,
		"JeM": JeM,
		"JoM": JoM,
		"njm": njm
	}


def check_normalization_condition_linear_rotor(output_file_path, basis_description_text, basis_function_matrix_data, normalization_matrix_data, total_number_of_basis_functions, quantum_numbers_data_list, deviation_tolerance_value, file_write_mode="new"):
	"""
	Checks whether the normalization condition <JM|J'M'> ≈ δ_JJ'MM' holds.
	
	- Logs shape information of the basis function matrix and normalization matrix.
	- Uses a numerical check to verify if the normalization matrix is close to the identity matrix.
	- Identifies and logs only significant deviations from the expected values.
	- Computes eigenvalues for numerical stability analysis.
	
	Arguments:
		output_file_path (str): The path to save the normalization check log file.
		basis_description_text (str): A textual description of the basis used.
		basis_function_matrix_data (numpy.ndarray): The numerical matrix representing the basis functions.
		normalization_matrix_data (numpy.ndarray): The numerical matrix <JM|J'M'> representing normalization.
		total_number_of_basis_functions (int): The total number of basis functions used in the system.
		quantum_numbers_data_list (numpy.ndarray): The array containing quantum number pairs for each basis function.
		deviation_tolerance_value (float): The threshold value for detecting deviations from the expected identity matrix.
		file_write_mode (str): Either "new" to overwrite the file or "append" to add data to an existing file.
	"""
	# Determine whether to overwrite or append to the output file
	file_open_mode = "w" if file_write_mode == "new" else "a"

	# Compute the deviation of the normalization matrix from the identity matrix
	identity_matrix_reference = np.eye(total_number_of_basis_functions)
	deviation_from_identity_matrix = np.abs(normalization_matrix_data - identity_matrix_reference)
	maximum_deviation_value = np.max(deviation_from_identity_matrix)

	with open(output_file_path, file_open_mode) as output_file:
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
			output_file.write("⚠ WARNING: Some eigenvalues deviate significantly from the expected value of 1!\n")

		output_file.write("=" * 80 + "\n")

	return maximum_deviation_value

def plot_heatmap(normalization_matrix_data, title):
	"""
	Visualizes the normalization matrix using a heatmap.
	
	Parameters:
		normalization_matrix_data (ndarray): The normalization matrix.
		basis_type (str): The type of basis function used.
		title (str): Custom title for the heatmap.
	"""
	if normalization_matrix_data is None or normalization_matrix_data.size == 0:
		print("Error: The normalization matrix is empty or None!")
		return

	#plt.style.use("dark_background")  # Set dark theme for better contrast
	plt.figure(figsize=(8, 6))

	"""
	sns.heatmap(
		np.abs(normalization_matrix_data),
		annot=False,
		cmap="plasma",  # High-contrast colormap on dark background
		linewidths=0.5,
		cbar=True
	)
	"""

	sns.heatmap(
		np.abs(normalization_matrix_data),
		annot=True,
		#cmap="Greys",  # Monochrome grayscale
		cmap="viridis",  # Monochrome grayscale
		linewidths=0.01,
		cbar=True
	)

	plt.title(f"{title}", fontsize=12, fontweight='bold')
	plt.xlabel("Basis Index", fontsize=10)
	plt.ylabel("Basis Index", fontsize=10)

	# Show the plot
	plt.show()

# Example usage:
# plot_heatmap(normalization_matrix_data, title)


def check_unitarity(file_name, basis_type, umat, small=1e-10, mode="new"):
	"""
	Checks whether the transformation matrix U satisfies the unitarity condition U U† = I.

	Parameters:
		file_name (str): Output file name for writing results.
		basis_type (str): Type of basis function.
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
		f.write(f"Checking Unitarity Condition for {basis_type} Transformation Matrix U\n")
		f.write("="*80 + "\n")
		f.write(f"Shape of {basis_type} U matrix: {umat.shape}\n")
		f.write(f"Max deviation from identity: {deviation}\n")

		if deviation < small:
			f.write("✔ The matrix U satisfies the unitarity condition.\n")
		else:
			f.write("⚠ WARNING: The matrix U does NOT satisfy the unitarity condition.\n")

	return deviation < small  # Returns True if unitarity holds, False otherwise


def compute_rotational_kinetic_energy_loop(umat, quantum_numbers_data_list, Bconst):
	"""
	Computes the rotational kinetic energy operator T_rot using a loop.

	Parameters:
	umat : numpy.ndarray
		Unitary normalization matrix (n_basis, n_basis).
	quantum_numbers_data_list : numpy.ndarray
		Array of quantum numbers where column 0 contains J values.
	Bconst : float
		Rotational constant.

	Returns:
	numpy.ndarray
		Rotational kinetic energy operator matrix T_rot.
	"""
	n_basis = umat.shape[0]
	T_rot = np.zeros((n_basis, n_basis), dtype=complex)  # Ensure the matrix can hold complex values

	# Compute rotational energy levels B * J(J+1)
	rotational_energies = Bconst * quantum_numbers_data_list[:, 0] * (quantum_numbers_data_list[:, 0] + 1)

	for jm in range(n_basis):
		for jmp in range(n_basis):
			sum_value = 0.0
			for s in range(n_basis):
				sum_value += umat[s, jm].conj() * umat[s, jmp] * rotational_energies[s]
			T_rot[jm, jmp] = sum_value

	#return np.real(T_rot)
	return T_rot


def compute_rotational_kinetic_energy_matrix(umat, quantum_numbers_data_list, Bconst):
	"""
	Computes the rotational kinetic energy operator T_rot using efficient matrix operations.

	Parameters:
	umat : numpy.ndarray
		Unitary normalization matrix (n_basis, n_basis).
	quantum_numbers_data_list : numpy.ndarray
		Array of quantum numbers where column 0 contains J values.
	Bconst : float
		Rotational constant.

	Returns:
	numpy.ndarray
		Rotational kinetic energy operator matrix T_rot.
	"""
	n_basis = umat.shape[0]

	# Validate Inputs
	if umat.shape[0] != umat.shape[1]:
		raise ValueError("Unitary normalization matrix U must be square.")
	if len(quantum_numbers_data_list) != n_basis:
		raise ValueError("Length of quantum numbers must match the dimensions of U.")

	# Compute rotational energy levels B * J(J+1)
	J_values = quantum_numbers_data_list[:, 0]
	rotational_energies = Bconst * J_values * (J_values + 1)

	# Create a diagonal matrix from rotational energies
	E_diag = np.diag(rotational_energies)

	# Compute T_rot using matrix multiplication
	T_rot = umat.conj().T @ E_diag @ umat

	#return np.real(T_rot)  # Return the real part of the resulting matrix
	return T_rot  # Return the real part of the resulting matrix


def compute_rotational_kinetic_energy_einsum(umat, quantum_numbers_data_list, Bconst, debug=False):
	"""
	Computes the rotational kinetic energy operator T_rot using efficient Einstein summation.

	Parameters:
	umat : numpy.ndarray
		Unitary normalization matrix (n_basis, n_basis).
	quantum_numbers_data_list : numpy.ndarray
		Array of quantum numbers where column 0 contains J values.
	Bconst : float
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
	if len(quantum_numbers_data_list) != n_basis:
		raise ValueError("Length of quantum numbers must match the dimensions of U.")

	# Extract J values (rotational quantum numbers)
	J_values = quantum_numbers_data_list[:, 0]

	# Compute the rotational energy levels B * J(J+1)
	rotational_energies = Bconst * J_values * (J_values + 1.0)

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


def check_hermiticity(H, matrix_name="H", tol=1e-10, debug=True, visualize=False):
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
		print("\n===== Hermiticity Check =====")
		print(f"✅ Matrix shape: {H.shape}")
		print(f"✅ Max deviation: {max_diff:.2e}")
		print(f"✅ Frobenius norm: {norm_diff:.2e}")
		print(f"✅ Mean deviation: {mean_diff:.2e}")

		if discrepancies:
			print(f"\n❌ {len(discrepancies)} discrepancies found (threshold = {tol}):")
			print("   Index (Row, Col)  | Deviation |")
			print("   --------------------------------")
			for i, j, value in discrepancies[:10]:  # Show first 10 discrepancies
				print(f"   ({i:3d}, {j:3d})   | {value:.2e}")
			if len(discrepancies) > 10:
				print("   ... (truncated)")

		else:
			print("✅ No discrepancies found. Matrix is Hermitian.")

	if visualize:
		fig, axes = plt.subplots(1, 2, figsize=(18, 5))
		#sns.heatmap(H.real, cmap="coolwarm", annot=False, ax=axes[0])
		sns.heatmap(H.real, cmap="viridis", linewidths=0.1, annot=True, ax=axes[0])
		axes[0].set_title(f"Original Matrix (Re[{matrix_name}])")

		#sns.heatmap(H_dagger.real, cmap="coolwarm", annot=False, ax=axes[1])
		sns.heatmap(H_dagger.real, cmap="viridis", linewidths=0.1, annot=True, ax=axes[1])
		axes[1].set_title(f"Hermitian Conjugate (Re[{matrix_name}†])")

		plt.tight_layout()
		plt.show()

		fig, ax = plt.subplots(figsize=(12, 7))
		#sns.heatmap(diff, cmap="viridis", annot=True, fmt=".2e", ax=ax)
		sns.heatmap(diff, cmap="viridis", linewidths=0.1, annot=True, fmt=".2e", ax=ax)
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
	print(f"\n{'J':<5}{'Energy (Kelvin)':>15}")
	print("=" * 20)
	for J, E in energies.items():
		print(f"{J:<5}{E:>15.2f}")
	
	return energies

def plot_rotational_levels(energies):
	"""
	Plots the rotational energy levels.
	
	Parameters:
	- energies (dict): Dictionary with J values as keys and energy values.
	"""
	J_values = list(energies.keys())
	energy_values = list(energies.values())
	
	plt.figure(figsize=(8, 5))
	plt.scatter(J_values, energy_values, color='b', label="Rotational Levels")
	plt.plot(J_values, energy_values, linestyle="dashed", color="gray")
	
	# Annotate energy values
	for J, E in energies.items():
		plt.text(J, E + 10, f"{E:.1f}", ha='center', fontsize=10)

	plt.xlabel("Rotational Quantum Number (J)")
	plt.ylabel("Energy (Kelvin)")
	plt.title("Rotational Energy Levels of a Rigid Rotor")
	plt.grid(True, linestyle="--", alpha=0.6)
	plt.legend()
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

def display_rotational_energies(diagonal_elements, quantum_numbers_data_list, Bconst):
	"""
	Displays the extracted diagonal elements as rotational energy levels.

	Parameters:
	- diagonal_elements (np.ndarray): Extracted diagonal elements representing energy levels.
	- quantum_numbers_data_list (np.ndarray): Array of quantum numbers, where each row represents a state
											  and the first column contains the J values (rotational quantum numbers).
	- Bconst (float): The rotational constant (cm⁻¹), used to compute rotational energy levels.

	Returns:
	None
	"""
	print("\nRotational Energy Levels")
	print("=" * 80)
	print(f"{'Quantum State (J)':^25} {'BJ(J+1) (Kelvin)':^25} {'<JM|T|JM> (Kelvin)':^25}")
	print("=" * 80)

	# Extracting J values from the quantum numbers data
	J_values = quantum_numbers_data_list[:, 0]

	# Compute the rotational energy levels B * J(J+1)
	for J, energy in zip(J_values, diagonal_elements):
		# Calculate the theoretical energy level based on the B constant
		theoretical_energy = Bconst * J * (J + 1)
		
		
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

def save_eigenvalues_eigenvectors_netcdf(H_rot, scaling_factor, filename="eigen_data.nc"):
	"""
	Computes, sorts, and saves eigenvalues and eigenvectors to a compressed NetCDF file.

	Parameters:
	- H_rot (ndarray): Rotational Hamiltonian matrix (NxN).
	- scaling_factor (float): Scaling factor for eigenvalues (unit conversion).
	- filename (str): Output NetCDF file name (default: "eigen_data.nc").

	Saves:
	- Eigenvalues (sorted) and their scaled versions as a 2-column variable.
	- Corresponding eigenvectors in sorted order.
	"""
	# Compute eigenvalues and eigenvectors efficiently
	eigenvalues, eigenvectors = eigh(H_rot)

	# Sort eigenvalues and rearrange eigenvectors accordingly
	sorted_indices = np.argsort(eigenvalues)
	sorted_eigenvalues = eigenvalues[sorted_indices]
	sorted_eigenvectors = eigenvectors[:, sorted_indices]

	# Stack eigenvalues and their scaled versions into a 2D array
	eigenvalue_matrix = np.column_stack((sorted_eigenvalues, sorted_eigenvalues / scaling_factor))

	# Create and write to NetCDF file
	with Dataset(filename, "w", format="NETCDF4") as ncfile:
		# Define dimensions
		N = H_rot.shape[0]
		ncfile.createDimension("N", N)
		ncfile.createDimension("eigen_components", 2)  # For eigenvalues and scaled values

		# Create variables with compression
		ev_var = ncfile.createVariable("eigenvalues", "f8", ("N", "eigen_components"), zlib=True)
		eigvec_var = ncfile.createVariable("eigenvectors", "f8", ("N", "N"), zlib=True)

		# Store data
		ev_var[:, :] = eigenvalue_matrix
		eigvec_var[:, :] = sorted_eigenvectors

	print(f"Eigenvalues and eigenvectors saved in {filename} (compressed NetCDF4)")


def compute_sorted_eigenvalues_and_eigenvectors(H_rot, scaling_factor):
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
	eigenvalue_matrix = np.column_stack((sorted_eigenvalues, sorted_eigenvalues / scaling_factor))

	return eigenvalue_matrix, sorted_eigenvectors


def save_eigenvalues_eigenvectors_netcdf(eigenvalue_matrix, sorted_eigenvectors, filename="eigen_data.nc"):
	"""
	Saves sorted eigenvalues and eigenvectors (real & imaginary) to a compressed NetCDF file.

	Parameters:
	- eigenvalue_matrix (ndarray): Nx2 matrix with sorted eigenvalues and their scaled versions.
	- sorted_eigenvectors (ndarray): NxN matrix of sorted eigenvectors (complex).
	- filename (str): Output NetCDF file name (default: "eigen_data.nc").
	"""
	N = sorted_eigenvectors.shape[0]  # Matrix size

	with Dataset(filename, "w", format="NETCDF4") as ncfile:
		# Define dimensions
		ncfile.createDimension("N", N)
		ncfile.createDimension("eigen_components", 2)  # For eigenvalues and scaled values

		# Create variables
		ev_var = ncfile.createVariable("eigenvalues", "f8", ("N", "eigen_components"), zlib=True)
		eigvec_real_var = ncfile.createVariable("eigenvectors_real", "f8", ("N", "N"), zlib=True)
		eigvec_imag_var = ncfile.createVariable("eigenvectors_imag", "f8", ("N", "N"), zlib=True)

		# Store real and imaginary parts separately
		ev_var[:, :] = eigenvalue_matrix
		eigvec_real_var[:, :] = sorted_eigenvectors.real
		eigvec_imag_var[:, :] = sorted_eigenvectors.imag

	print(f"✅ Eigenvalues and eigenvectors saved in {filename} (compressed NetCDF4)")


def load_eigenvalues_eigenvectors_netcdf(filename="eigen_data.nc"):
	"""
	Reads eigenvalues and eigenvectors from a NetCDF file.

	Parameters:
	- filename (str): Name of the NetCDF file (default: "eigen_data.nc").

	Returns:
	- eigenvalues (ndarray): Sorted eigenvalues (1D array).
	- scaled_eigenvalues (ndarray): Scaled eigenvalues (1D array).
	- eigenvectors (ndarray): Corresponding sorted eigenvectors (NxN matrix).
	"""
	try:
		with Dataset(filename, "r") as ncfile:
			# Check if the expected variables exist in the file
			if "eigenvalues" not in ncfile.variables:
				raise KeyError("Missing 'eigenvalues' variable in NetCDF file.")
			
			eigenvalue_matrix = ncfile.variables["eigenvalues"][:]
			eigenvalues = eigenvalue_matrix[:, 0]  # First column: original eigenvalues
			scaled_eigenvalues = eigenvalue_matrix[:, 1]  # Second column: scaled values
			
			# Handling complex eigenvectors if stored separately
			if "eigenvectors_real" in ncfile.variables and "eigenvectors_imag" in ncfile.variables:
				eigvec_real = ncfile.variables["eigenvectors_real"][:]
				eigvec_imag = ncfile.variables["eigenvectors_imag"][:]
				eigenvectors = eigvec_real + 1j * eigvec_imag  # Reconstruct complex matrix
			elif "eigenvectors" in ncfile.variables:
				eigenvectors = ncfile.variables["eigenvectors"][:]
			else:
				raise KeyError("Missing eigenvectors in NetCDF file.")

		return eigenvalues, scaled_eigenvalues, eigenvectors

	except FileNotFoundError:
		print(f"❌ Error: File '{filename}' not found.")
		return None, None, None
	except Exception as e:
		print(f"❌ Error: {e}")
		return None, None, None

def debug_eigenvalues_eigenvectors(H_rot, sorted_eigenvalues, sorted_eigenvectors):
	"""
	Debugs and verifies the correctness of computed eigenvalues and eigenvectors.

	Parameters:
	- H_rot (ndarray): The original rotational Hamiltonian matrix.
	- sorted_eigenvalues (ndarray): The computed sorted eigenvalues.
	- sorted_eigenvectors (ndarray): The computed sorted eigenvectors.

	Returns:
	- None: Prints debugging information and raises assertion errors if checks fail.
	"""

	print("\n🔍 DEBUGGING EIGENVALUES & EIGENVECTORS 🔍")

	# 1️⃣ Check if H_rot is Hermitian (Symmetric for Real Case)
	assert np.allclose(H_rot, H_rot.T.conj()), "❌ H_rot is not Hermitian (symmetric for real case)."
	print("✅ H_rot is Hermitian.")

	# 2️⃣ Verify Eigenvalue Computation
	print("\n🔹 Computed Eigenvalues:\n", sorted_eigenvalues)
	assert np.all(sorted_eigenvalues[:-1] <= sorted_eigenvalues[1:]), "❌ Eigenvalues are not properly sorted!"
	print("✅ Eigenvalues are sorted correctly.")

	# 3️⃣ Check the Eigenvectors' Orthogonality (Eigenvectors should be orthonormal)
	identity_check = np.dot(sorted_eigenvectors.T.conj(), sorted_eigenvectors)
	print("\n🔹 Orthogonality Check (Should be Identity Matrix):\n", identity_check)
	assert np.allclose(identity_check, np.eye(identity_check.shape[0])), "❌ Eigenvectors are not orthonormal!"
	print("✅ Eigenvectors are orthonormal.")

	"""
	# 4️⃣ Validate Eigenvalue Equation (HΨ = EΨ)
	reconstructed_H = sorted_eigenvectors @ np.diag(sorted_eigenvalues) @ sorted_eigenvectors.T.conj()
	print("\n🔹 Reconstructed H_rot from Eigenvalues and Eigenvectors:\n", reconstructed_H)
	assert np.allclose(reconstructed_H, H_rot), "❌ Eigenvalue equation validation failed!"
	print("✅ Eigenvalue equation holds (HΨ = EΨ).")
	"""

	# 5️⃣ Debugging the NetCDF Storage Issue
	if np.iscomplexobj(sorted_eigenvectors):
		print("⚠️ Warning: Eigenvectors contain complex numbers!")
		sorted_eigenvectors = sorted_eigenvectors.real  # Store only the real part if justified
		print("🔹 Only real part of eigenvectors will be stored in NetCDF.")

	print("\n🎯 ✅ All checks passed! Eigenvalues and eigenvectors are computed correctly. 🎯")

# eigenvalues_matrix, sorted_eigenvectors = compute_sorted_eigenvalues_and_eigenvectors(H_rot, 1.0)
# sorted_eigenvalues = eigenvalues_matrix[:, 0]  # Extract original eigenvalues
# debug_eigenvalues_eigenvectors(H_rot, sorted_eigenvalues, sorted_eigenvectors)


def main():
	# Parse command-line arguments
	args = parse_arguments()
	potential_strength = args.potential_strength
	max_angular_momentum = args.max_angular_momentum
	spin_state = args.spin

	# No. of grid points along theta and phi
	theta_grid_count = int(2 * max_angular_momentum + 17)
	phi_grid_count = int(2 * theta_grid_count + 5)

	# Tolerance limit for a harmitian matrix
	deviation_tolerance_value = 10e-12

	# print the normalization
	io_write = True
	normalization_check = True
	unitarity_check = True
	pot_write = False
	kinetic_energy_operator_hermiticity_test = False

	Bconst = 60.853  # cm-1 Taken from NIST data https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000
	CMRECIP2KL = 1.4387672	   	# cm^-1 to Kelvin conversion factor
	Bconst = Bconst * CMRECIP2KL

	#energies = rotational_energy_levels(Bconst, 10)
	#plot_rotational_levels(energies)

	# Display input parameters
	show_simulation_details(potential_strength, max_angular_momentum, spin_state, theta_grid_count, phi_grid_count)

	prefix = "output_file_for_checking_orthonormality_condition"
	basis_type, file_name = generate_filename(spin_state, max_angular_momentum, potential_strength, theta_grid_count, phi_grid_count, prefix)

	# Separator line
	print("\n**")
	print(colored("basis_type".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{basis_type}".ljust(VALUE_WIDTH), VALUE_COLOR))

	# Gauss-Quadrature points
	xGL, wGL, phixiGridPts, dphixi = compute_legendre_quadrature(theta_grid_count, phi_grid_count, io_write)

	# Generate (J, M) matrices for each nuclear spin isomer type
	quantum_numbers_data_list = bfunc.generate_linear_rotor_quantum_numbers(max_angular_momentum, "spinless")
	
	df = pd.DataFrame(quantum_numbers_data_list, columns=["J", "M"])
	# Separator line
	print("\n**")
	print(colored(f"All quantum numbers\n".ljust(LABEL_WIDTH), LABEL_COLOR))
	print(df)

	# Generate (J, M) matrices for each nuclear spin isomer type
	quantum_numbers_data_list_for_spin_state = bfunc.generate_linear_rotor_quantum_numbers(max_angular_momentum, spin_state)
	
	df = pd.DataFrame(quantum_numbers_data_list_for_spin_state, columns=["J", "M"])
	# Separator line
	print("\n**")
	print(colored(f"Quantum numbers for {spin_state} isomer.\n".ljust(LABEL_WIDTH), LABEL_COLOR))
	print(df)

	# njm, JM, JeM, JoM = compute_basis_functions(max_angular_momentum, spin_state)
	basis_functions_info = get_number_of_basis_functions_by_spin_states(max_angular_momentum, spin_state)
	total_number_of_states = basis_functions_info["JM"]
	total_number_of_spin_states = basis_functions_info["njm"]
	
	# Real spherical harmonics basis <cos(θ), φ | JM> as a 2D matrix 'basisfun_real' with shape (theta_grid_count * phi_grid_count, n_basis), 
	# where each column corresponds to a unique (J, M) quantum number pair and rows map to grid points across θ and φ angles.
	n_basis_real = total_number_of_states
	basisfun_real = bfunc.spherical_harmonicsReal(n_basis_real, theta_grid_count, phi_grid_count, quantum_numbers_data_list, xGL, wGL, phixiGridPts, dphixi)
	# Separator line
	print("\n**")
	print(colored("shape of ", INFO_COLOR) + colored("spherical_harmonicsReal or basisfun_real: ".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{basisfun_real.shape}".ljust(VALUE_WIDTH), VALUE_COLOR))

	if (normalization_check):
		# Compute the overlap (normalization) matrix to check if the basis functions are orthonormal.  
		# The resulting normalization_matrix_data_real is of size (n_basis, n_basis), where n_basis is the number of basis functions.  
		# If the basis functions are perfectly normalized and orthogonal, normalization_matrix_data_real should be close to the identity matrix.  
		normalization_matrix_data_real = np.einsum('ij,ik->jk', np.conjugate(basisfun_real), basisfun_real)  # (n_points, n_basis) x (n_points, n_basis) → (n_basis, n_basis)
		#normalization_matrix_data_real = np.tensordot(basisfun_real, np.conjugate(basisfun_real), axes=([0], [0]))
		#df = pd.DataFrame(normalization_matrix_data_real)
		#print(df)
		
		#
		print(colored("shape of ", INFO_COLOR) + colored("normalization_matrix_data_real: ".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{normalization_matrix_data_real.shape}".ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")
		print(colored("**\nFile name for checking orthonormality condition: ".ljust(LABEL_WIDTH), LABEL_COLOR) + f"{file_name}" + "\n")

		output_file_path = file_name 
		basis_description_text = "Real Spherical Harmonics Basis |JM> For A Linear Rotor"
		check_normalization_condition_linear_rotor(
			output_file_path,
			basis_description_text,
			basisfun_real,
			normalization_matrix_data_real,
			n_basis_real,
			quantum_numbers_data_list,
			deviation_tolerance_value,
			file_write_mode="new"
		)
		title = f"Heatmap of normalization matrix \n Real basis"
		plot_heatmap(normalization_matrix_data_real, title)

		is_Hermitian, max_diff = check_hermiticity(normalization_matrix_data_real, "S", tol=1e-10, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")


	n_basis_complex = total_number_of_states
	# Construction of complex basis functions 
	basisfun_complex = bfunc.spherical_harmonicsComp(n_basis_complex, theta_grid_count, phi_grid_count, quantum_numbers_data_list, xGL, wGL, phixiGridPts, dphixi)
	if (normalization_check):
		# Orthonormality test for "complex basis"
		normalization_matrix_data_complex = np.einsum('ij,ik->jk', np.conjugate(basisfun_complex), basisfun_complex)  # (n_points, n_basis) x (n_points, n_basis) → (n_basis, n_basis)
		basis_description_text = "Complex Spherical Harmonics Basis |JM> For A Linear Rotor"
		check_normalization_condition_linear_rotor(
			output_file_path,
			basis_description_text,
			basisfun_complex,
			normalization_matrix_data_complex,
			n_basis_complex,
			quantum_numbers_data_list,
			deviation_tolerance_value,
			file_write_mode="append"
		)
		title = f"Heatmap of normalization matrix \n Complex basis"
		plot_heatmap(normalization_matrix_data_complex, title)

		is_Hermitian, max_diff = check_hermiticity(normalization_matrix_data_complex, "S", tol=1e-10, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

	#
	# Construction of Unitary Matrix 
	# umat = np.tensordot(np.conjugate(basisfun_complex), basisfun_real, axes=([0], [0]))
	umat = np.einsum('ij,ik->jk', np.conjugate(basisfun_complex), basisfun_real)
	#umat = basisfun_complex.conj().T @ basisfun_real
	whoami()

	if (unitarity_check):
		check_unitarity(file_name, basis_type, umat, mode="append")
		# Compute UU†
		umat_unitarity = np.einsum('ia,ja->ij', umat, np.conjugate(umat))
		#umat_unitarity = np.einsum('ia,ja->ij', umat, umat.conj())
		#umat_unitarity = np.einsum('ia,ib->ab', umat, umat.conj())
		#umat_unitarity = umat.conj().T @ umat
		title = f"Heatmap of UU† matrix for {spin_state} spin state"
		plot_heatmap(umat_unitarity, title)

		is_Hermitian, max_diff = check_hermiticity(umat_unitarity, "(UU†)", tol=1e-40, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

	# Call the function to compute the rotational kinetic energy operator
	#T_rot_einsum = compute_rotational_kinetic_energy_einsum(umat, quantum_numbers_data_list, Bconst)
	T_rot_einsum = compute_rotational_kinetic_energy_einsum(umat, quantum_numbers_data_list, Bconst, debug=False)
	# Extract and display rotational energies
	diagonal_energies = extract_diagonal(T_rot_einsum.real)
	display_rotational_energies(diagonal_energies, quantum_numbers_data_list, Bconst)

	if kinetic_energy_operator_hermiticity_test: 
		is_Hermitian, max_diff = check_hermiticity(T_rot_einsum, "T", tol=1e-10, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")

	if False:
		V_rot_1 = np.einsum('ia,ja->ij', umat, umat.conj())
		V_rot_2 = np.einsum('ia,ib->ab', umat, umat.conj())

		is_Hermitian, max_diff = check_hermiticity(V_rot, "V", tol=1e-10, debug=True, visualize=False)
	V_rot_einsum = compute_potential_energy_einsum(basisfun_complex, umat, xGL, theta_grid_count, phi_grid_count, potential_strength, debug=False)
	is_Hermitian, max_diff = check_hermiticity(V_rot_einsum, "V", tol=1e-10, debug=True, visualize=False)
	print(f"Is the matrix Hermitian? {is_Hermitian}")

	H_rot = T_rot_einsum + V_rot_einsum

	is_Hermitian, max_diff = check_hermiticity(H_rot, "H", tol=1e-10, debug=True, visualize=False)
	print(f"Is the matrix Hermitian? {is_Hermitian}")

	# Compute eigenvalues and eigenvectors
	eigenvalues_matrix, sorted_eigenvectors = compute_sorted_eigenvalues_and_eigenvectors(H_rot, CMRECIP2KL)

	# Debugging function call
	debug_eigenvalues_eigenvectors(H_rot, eigenvalues_matrix, sorted_eigenvectors)
	whoami()

	# Print the results (example)
	print("Sorted Eigenvalues and their scaled versions:\n", eigenvalues_matrix)

	#print("Corresponding Sorted Eigenvectors:\n", sorted_eigenvectors)
	save_eigenvalues_eigenvectors_netcdf(eigenvalues_matrix, sorted_eigenvectors)

	# Example Usage
	eigenvalues, scaled_eigenvalues, eigenvectors = load_eigenvalues_eigenvectors_netcdf("eigen_data.nc")



"""
	for idx in range(4):
		eigVecRe = np.real(np.dot(np.conjugate(
			eigVec_sort[:, idx].T), eigVec_sort[:, idx]))
		eigVecIm = np.imag(np.dot(np.conjugate(
			eigVec_sort[:, idx].T), eigVec_sort[:, idx]))
		print(
			"Checking normalization of ground state eigenfunction - Re: " +
			str(eigVecRe) +
			" Im: " +
			str(eigVecIm))

		avgHpotL = np.dot(Hpot, eigVec_sort[:, idx])
		avgHpot = np.dot(np.conjugate(eigVec_sort[:, idx].T), avgHpotL)
		print("Expectation value of ground state potential - Re: " +
			  str(avgHpot.real) + " Im: " + str(avgHpot.imag))
	# printing block is closed

	# printing block is opened
	idx = 0
	avgHpotL = np.dot(Hpot, eigVec_sort[:, idx])
	avgHpot = np.dot(np.conjugate(eigVec_sort[:, idx].T), avgHpotL)

	gs_eng_file = prefile + "ground-state-energy-" + strFile
	gs_eng_write = open(gs_eng_file, 'w')
	gs_eng_write.write(
		"#Printing of ground state energies in inverse Kelvin - " + "\n")
	gs_eng_write.write('{0:1} {1:^19} {2:^20}'.format("#", "<T+V>", "<V>"))
	gs_eng_write.write("\n")
	gs_eng_write.write(
		'{0:^20.8f} {1:^20.8f}'.format(
			eigVal_sort[0], avgHpot.real))
	gs_eng_write.write("\n")
	gs_eng_write.close()
	# printing block is closed

	# computation of reduced density matrix
	reduced_density = np.zeros((njkm, max_angular_momentum + 1), dtype=complex)
	for i in range(njkm):
		for ip in range(njkm):
			if ((njkmQuantumNumList[i, 1] == njkmQuantumNumList[ip, 1]) and (
					njkmQuantumNumList[i, 2] == njkmQuantumNumList[ip, 2])):
				reduced_density[i, njkmQuantumNumList[ip, 0]] = np.conjugate(
					eigVec_sort[i, 0]) * eigVec_sort[ip, 0]

	gs_ang_file = prefile + "ground-state-theta-distribution-" + strFile
	gs_ang_write = open(gs_ang_file, 'w')
	gs_ang_write.write(
		"#Printing of ground state theta distribution - " + "\n")
	gs_ang_write.write(
		'{0:1} {1:^19} {2:^20}'.format(
			"#", "cos(theta)", "reduced density"))
	gs_ang_write.write("\n")

	sum3 = complex(0.0, 0.0)
	for th in range(theta_grid_count):
		sum1 = complex(0.0, 0.0)
		for i in range(njkm):
			for ip in range(njkm):
				if ((njkmQuantumNumList[i, 1] == njkmQuantumNumList[ip, 1]) and (
						njkmQuantumNumList[i, 2] == njkmQuantumNumList[ip, 2])):
					sum1 += 4.0 * math.pi * math.pi * \
						reduced_density[i, njkmQuantumNumList[ip, 0]] * dJKM[i, th] * dJKM[ip, th]
		gs_ang_write.write(
			'{0:^20.8f} {1:^20.8f}'.format(
				xGL[th], sum1.real / wGL[th]))
		gs_ang_write.write("\n")
		sum3 += sum1
	gs_ang_write.close()
	# printing block is closed

	print("Normalization: reduced density matrix = ", sum3)
"""

if __name__ == "__main__":
	main()
