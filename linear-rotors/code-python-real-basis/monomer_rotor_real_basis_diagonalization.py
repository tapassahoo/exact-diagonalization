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
		annot=False,
		cmap="Greys",  # Monochrome grayscale
		linewidths=0.3,
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


def compute_rotational_kinetic_energy_einsum(umat, quantum_numbers_data_list, Bconst):
	"""
	Computes the rotational kinetic energy operator T_rot using efficient Einstein summation.

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

	# Compute T_rot using Einstein summation notation for efficient matrix multiplication
	T_rot = np.einsum('ji, jk, kl -> il', umat.conj(), E_diag, umat)

	#return np.real(T_rot)  # Return the real part of the resulting matrix
	return T_rot  # Return the real part of the resulting matrix


def compute_potential_operator(basisfun_complex, umat, xGL, phi_grid_count, potential_strength):
	"""
	Computes the potential energy operator in the rotational eigenbasis.

	Parameters:
	- basisfun_complex: Complex spherical harmonics basis functions.
	- umat: Unitary transformation matrix from (l, m) to (J, M) basis.
	- xGL: Gauss-Legendre quadrature nodes (cos(theta)).
	- phi_grid_count: Number of azimuthal grid points.
	- potential_strength: Scaling factor A in V(theta) = -A cos(theta).

	Returns:
	- V_rot: Potential energy operator matrix in the |J, M⟩ basis.
	"""
	# Compute potential function on the grid
	pot_func_grid = -potential_strength * np.repeat(xGL, phi_grid_count)

	# Compute potential matrix in (l, m) basis
	H_pot = np.einsum("gi,g,gj->ij", basisfun_complex.conj(), pot_func_grid, basisfun_complex)

	# Transform to rotational basis
	V_rot = umat.conj().T @ H_pot @ umat

	return V_rot


def check_hermiticity(H, matrix_name, tol=1e-10, debug=True, visualize=False):
	"""
	Checks if a given matrix H is Hermitian and identifies discrepancies.

	Parameters:
	- H (np.ndarray): Input complex matrix.
	- tol (float): Tolerance for detecting non-Hermitian elements.
	- debug (bool): If True, prints debugging information.
	- visualize (bool): If True, generates heatmap plots.

	Returns:
	- bool: True if H is Hermitian, False otherwise.
	- list: List of (row, col, discrepancy value) for non-Hermitian elements.
	"""

	if not isinstance(H, np.ndarray):
		raise TypeError("Input matrix H must be a NumPy array.")

	if H.shape[0] != H.shape[1]:
		raise ValueError("Matrix H must be square.")

	H_dagger = H.conj().T  # Compute Hermitian conjugate (H†)
	diff = np.abs(H - H_dagger)  # Absolute difference |H - H†|

	max_diff = np.max(diff).item()  # Extract scalar value
	norm_diff = np.linalg.norm(diff)  # Frobenius norm of difference
	mean_diff = np.mean(diff)  # Mean deviation in Hermiticity

	discrepancy_indices = np.argwhere(diff > tol)  # Find discrepancies
	discrepancies = [(i, j, diff[i, j]) for i, j in discrepancy_indices]

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
				print(f"   ({i:3d}, {j:3d})	  | {value:.2e}")
			if len(discrepancies) > 10:
				print("   ... (truncated)")

		else:
			print("✅ No discrepancies found. Matrix is Hermitian.")

	if visualize:
		# First Frame
		fig, axes = plt.subplots(1, 2, figsize=(18, 5))
		sns.heatmap(H.real, cmap="coolwarm", annot=False, ax=axes[0])
		axes[0].set_title(f"Original Matrix (Re[{matrix_name}])")
		
		sns.heatmap(H_dagger.real, cmap="coolwarm", annot=False, ax=axes[1])
		axes[1].set_title(f"Hermitian Conjugate (Re[{matrix_name}†])")
		
		#sns.heatmap(diff, cmap="viridis", annot=True, fmt=".2e", ax=axes[2])
		#axes[2].set_title(f"Difference |H - H†| (Max: {max_diff:.2e})")

		plt.tight_layout()
		plt.show()
			
		# Second Frame
		fig, ax = plt.subplots(figsize=(12, 7))

		sns.heatmap(diff, cmap="viridis", annot=True, fmt=".2e", ax=ax)
		ax.set_title(f"Difference |{matrix_name} - {matrix_name}†| (Max: {max_diff:.2e})")

		# Label the x and y axes
		ax.set_xlabel("Basis Index")
		ax.set_ylabel("Basis Index")
	
		plt.tight_layout()
		plt.show()

	return is_hermitian, discrepancies


def main():
	# Parse command-line arguments
	args = parse_arguments()
	potential_strength = args.potential_strength
	max_angular_momentum = args.max_angular_momentum
	spin_state = args.spin

	# No. of grid points along theta and phi
	theta_grid_count = int(2 * max_angular_momentum + 7)
	phi_grid_count = int(2 * theta_grid_count + 5)

	# Tolerance limit for a harmitian matrix
	deviation_tolerance_value = 10e-12

	# print the normalization
	io_write = True
	normalization_check = True
	unitarity_check = True
	pot_write = False
	debugging = True

	Bconst = 60.853  # cm-1 Taken from NIST data https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000
	CMRECIP2KL = 1.4387672	   	# cm^-1 to Kelvin conversion factor
	Bconst = Bconst * CMRECIP2KL

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
	# umat = np.tensordot(np.conjugate(basisfun_complex), basisfunc_real, axes=([0], [0]))
	# umat = np.einsum('ij,ik->jk', np.conjugate(basisfun_complex), basisfun_real)
	umat = basisfun_complex.conj().T @ basisfun_real

	if (unitarity_check):
		check_unitarity(file_name, basis_type, umat, mode="append")
		# Compute UU†
		# umat_unitarity = np.einsum('ia,ja->ij', umat, np.conjugate(umat))
		umat_unitarity = umat.conj().T @ umat
		title = f"Heatmap of UU† matrix for {spin_state} spin state"
		plot_heatmap(umat_unitarity, title)

		is_Hermitian, max_diff = check_hermiticity(umat_unitarity, "UU", tol=1e-40, debug=True, visualize=True)
		print(f"Is the matrix Hermitian? {is_Hermitian}")


	whoami()
	# Calculate T_rot using both methods
	# T_rot_loop = compute_rotational_kinetic_energy_loop(umat, quantum_numbers_data_list, Bconst)

	# print("Rotational Kinetic Energy Operator T_rot (for loop):")
	# print(T_rot_loop)
	
	# Call the function to compute the rotational kinetic energy operator
	T_rot_matrix = compute_rotational_kinetic_energy_matrix(umat, quantum_numbers_data_list, Bconst)

	# Display the result
	# print("Rotational Kinetic Energy Operator T_rot (matrix):")
	# print(T_rot_matrix)

	# Call the function to compute the rotational kinetic energy operator
	# T_rot_einsum = compute_rotational_kinetic_energy_einsum(umat, quantum_numbers_data_list, Bconst)

	# Display the result
	# print("Rotational Kinetic Energy Operator T_rot (einsum):")
	# print(T_rot_einsum)

	# are_close = np.allclose(T_rot_loop, T_rot_matrix, T_rot_einsum) 

	# Check if they are close
	# print("\nAre the results close? ", are_close)

	# Example Usage
	#H_given = np.array([[2, 1j], [-1j, 3]])  # Example Hermitian matrix

	#is_Hermitian, max_diff = plot_hermiticity_analysis(T_rot_matrix, tol=1e-10, debug=True, plot_type="three")
	#print(f"Is the matrix Hermitian? {is_Hermitian}")
	#is_Hermitian, max_diff = plot_hermiticity_analysis(T_rot_matrix, tol=1e-10, debug=True, plot_type="one")
	#print(f"Maximum difference |H - H†|: {max_diff:.2e}")
	is_Hermitian, max_diff = check_hermiticity(T_rot_matrix, tol=1e-10, debug=True, visualize=True)
	print(f"Is the matrix Hermitian? {is_Hermitian}")
	whoami()

	# Compute potential function on the grid
	#pot_func_grid = -potential_strength * np.repeat(xGL, phi_grid_count)
	pot_func_grid = np.repeat(1.0, theta_grid_count*phi_grid_count)
	print(pot_func_grid)
	print(pot_func_grid.shape)
	H_pot = np.einsum("gi,g,gj->ij", basisfun_complex.conj(), pot_func_grid, basisfun_complex)
	# Transform to rotational basis
	#V_rot = umat.conj().T @ H_pot @ umat
	V_rot = umat.conj().T @ umat
	print(V_rot)
	df = pd.DataFrame(V_rot)
	print(df)
	is_Hermitian, max_diff = plot_hermiticity_analysis(V_pot, tol=1e-10, debug=True, plot_type="three")
	is_Hermitian, max_diff = plot_hermiticity_analysis(V_pot, tol=1e-10, debug=True, plot_type="one")
	print(f"Is the matrix Hermitian? {is_Hermitian}")
	print(f"Maximum difference |H - H†|: {max_diff:.2e}")
	whoami()

	# Example Usage
	#H_test = np.array([
	#	[2, 1j, 0.00000000001], 
	#	[-1j, 3, 0], 
	#	[0, 0, 4]
	#])  # Slight numerical error

	umat_unitarity = umat.conj().T @ umat
	is_Hermitian, discrepancy_list = check_hermiticity(umat_unitarity)

	if not is_Hermitian:
		print("\n🔴 Hermiticity Check Failed!")
	else:
		print("\n🟢 Matrix is Hermitian.")

	# Example Usage
	H_test = np.array([
		[2, 1j, 0.00000000001], 
		[-1j, 3, 0], 
		[0, 0, 4]
	])  # Slight numerical error

	is_Hermitian, discrepancy_list = check_hermiticity(H_test)

	if not is_Hermitian:
		print("\n🔴 Hermiticity Check Failed!")
	else:
		print("\n🟢 Matrix is Hermitian.")


	whoami()


	V_rot_matrix = compute_potential_operator(basisfun_complex, umat, xGL, phi_grid_count, potential_strength)
	#df=pd.DataFrame(V_rot_matrix)
	#is_Hermitian, max_diff = check_hermiticity_visual(V_rot_matrix)
	#print(f"Is the matrix Hermitian? {is_Hermitian}")
	#print(f"Maximum difference |H - H†|: {max_diff:.2e}")
	#print(df)

	whoami()

"""
	Htot = Hrot1 + Hpot
	if (np.all(np.abs(Htot - Htot.T) < deviation_tolerance_value) == False):
		print("The Hamiltonian matrx Htot is not hermitian.")
		exit()

	# Estimation of eigenvalues and eigenvectors begins here
	eigVal, eigVec = LA.eigh(Htot)
	# prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
	sortIndex_eigVal = eigVal.argsort()
	eigVal_sort = eigVal[sortIndex_eigVal]
	eigVec_sort = eigVec[:, sortIndex_eigVal]
	# Estimation of eigenvalues and eigenvectors ends here

	# printing block is opened
	eigVal_comb = np.array([eigVal_sort, eigVal_sort / CMRECIP2KL])

	eigVal_file = prefile + "eigen-values-" + strFile
	np.savetxt(
		eigVal_file,
		eigVal_comb.T,
		fmt='%20.8f',
		delimiter=' ',
		header='Energy levels of a aymmetric top - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')
	exit()

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
