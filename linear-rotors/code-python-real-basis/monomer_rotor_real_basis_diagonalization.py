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
#	   Example:																					  #
#	   python monomer_rotor_real_basis_diagonalization.py 10.0 2 spinless						  #
#																								  #
# ------------------------------------------------------------------------------------------------#
#																								  #
# Inputs:																						  #
#	   a) Potential strength = strength															  #
#	   b) Highest value of Angular quantum number = Jmax										  #
#	   c) Specification of spin isomer = spin_isomer											  #
#																								  #
# Outputs: Eigenvalues and eigenfunctions														  #
#																								  #
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
	"""Parse command-line arguments for potential strength, Jmax, and spin isomer."""

	# Initialize parser for command-line arguments
	parser = argparse.ArgumentParser(
		prog="monomer_rotor_real_basis_diagonalization.py",
		description="Computation of Eigenvalues and Eigenfunctions of a Linear Rotor Using Real Spherical Harmonics.",
		epilog="Enjoy the program! :)")

	# Define the arguments with clear help messages and types
	parser.add_argument(
		"strength",
		type=float,
		help="Interaction strength of the potential in the form A*cos(theta). Enter a real number."
	)

	parser.add_argument(
		"jmax",
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
	print ('*'*80)
	print("\nATTENTION: \n")
	print("%s/%s%s" %("The function is \n" + sys._getframe(1).f_code.co_filename, sys._getframe(1).f_code.co_name, "\nand the line number is " + str(sys._getframe(1).f_lineno)))
	print("")
	print ('*'*80)
	exit()


def display_parameters(strength, jmax, spin_isomer, size_theta, size_phi):
	"""Display the input parameters for the diagonalization."""

	now = datetime.now()  # current date and time
	date_time = now.strftime("%d/%m/%Y, %H:%M:%S")

	user_name = getpass.getuser()
	input_dir_path = os.getcwd()
	home = os.path.expanduser("~")

	# Separator line
	print(colored("*" * 80, SEPARATOR_COLOR))

	# Date and time section
	print(colored("Date and time:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(date_time.ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")

	# Debugging information
	print(colored("Debug mode is enabled.", DEBUG_COLOR) + "\n")
	print("**")
	print(colored("File systems are given below:", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("user_name:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(user_name.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("home:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(home.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("input_dir_path:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(input_dir_path.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Location of pkg_basis_func_rotors.basis_func_rotors as bfunc:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(bfunc.__file__.ljust(VALUE_WIDTH), VALUE_COLOR))
	print("\n**")

	# Input parameters section
	print(colored("Input parameters", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Potential strength (strength):".ljust(LABEL_WIDTH), INFO_COLOR) + colored(str(strength).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Highest Angular Quantum Number (jmax):".ljust(LABEL_WIDTH), INFO_COLOR) + colored(f"{jmax}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Spin Isomer:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(f"{spin_isomer}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print("\n**")

	# Additional grid information
	print(colored("Grid Information", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Number of theta grids:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(f"{size_theta}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Number of phi and chi grids:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(f"{size_phi}".ljust(VALUE_WIDTH), VALUE_COLOR))


def generate_filename(
	spin_isomer: str, 
	jmax: int, 
	strength: float, 
	size_theta: int, 
	size_phi: int, 
	prefix: Optional[str] = ""
) -> str:
	"""
	Generates a descriptive filename based on parameters for a linear rotor system.

	Parameters:
	- spin_isomer (str): The spin isomer type ("spinless", "para", or "ortho").
	- jmax (int): Highest angular quantum number.
	- strength (float): Field strength in Kelvin.
	- size_theta (int): Number of theta grids.
	- size_phi (int): Number of phi grids.
	- prefix (str, optional): Directory path or prefix for the file name. Defaults to "".

	Returns:
	- str: Constructed file name.
	"""
	# Determine isomer and basis type based on spin isomer
	if spin_isomer == "spinless":
		isomer = "spinless"
		basis_type = "none"
	elif spin_isomer == "para":
		isomer = "para"
		basis_type = "even"
	elif spin_isomer == "ortho":
		isomer = "ortho"
		basis_type = "odd"
	else:
		raise ValueError("Unknown spin isomer type: expected 'spinless', 'para', or 'ortho'.")

	# Construct the file name in a logical, readable format
	filename = (
		f"{prefix}_H2_{isomer}_isomer_jmax{jmax}_"
		f"strength{strength}K_"
		f"grids_theta{size_theta}_phi{size_phi}_diag.txt"
	)
	
	return filename


def compute_legendre_quadrature(size_theta, size_phi, io_write):
	"""
	Computes Gaussian quadrature points and weights for Legendre polynomials,
	along with phi grid points for a specified number of theta and phi points.

	Parameters:
	- size_theta (int): Number of theta points for Legendre quadrature.
	- size_phi (int): Number of phi points for uniform grid.
	- io_write (bool): If True, prints the quadrature points and weights.

	Returns:
	- xGL (np.ndarray): Gaussian quadrature points.
	- wGL (np.ndarray): Corresponding weights.
	- phixiGridPts (np.ndarray): Uniformly spaced phi grid points.
	- dphixi (float): Phi grid spacing.
	"""
	# Compute Gaussian quadrature points and weights
	xGL, wGL = np.polynomial.legendre.leggauss(size_theta)
	phixiGridPts = np.linspace(0, 2 * np.pi, size_phi, endpoint=False)
	dphixi = 2. * np.pi / size_phi

	# Optionally print the quadrature points and weights
	if io_write:
		print("|" + "-" * 48 + "|")
		print("| Gaussian Quadrature for Legendre Polynomials")
		print("|")
		print("| Quadrature Points (xGL):")
		print(xGL)
		print("|")
		print("| Corresponding Weights (wGL):")
		print(wGL)
		print("|" + "-" * 48 + "|")
		sys.stdout.flush()

	return xGL, wGL, phixiGridPts, dphixi


def get_number_of_basis_functions_by_spin_isomers(jmax, spin_isomer):
	"""
	Gets and displays the number of basis functions for a linear rotor
	categorized by spin isomers (spinless, para, ortho).

	Parameters:
	- jmax (int): The highest angular quantum number.
	- spin_isomer (str): The spin isomer type ("spinless", "para", or "ortho").

	Returns:
	- dict: A dictionary with JM, JeM, JoM, and njm values.
	"""
	# Calculate the total number of basis functions
	JM = int((jmax + 1)**2)

	# Determine the even (JeM) and odd (JoM) basis function counts
	if (jmax % 2) == 0:
		JeM = int((JM + jmax + 1) / 2)
		JoM = JM - JeM
	else:
		JoM = int((JM + jmax + 1) / 2)
		JeM = JM - JoM

	# Assign njm based on the spin isomer
	if spin_isomer == "spinless":
		njm = JM
	elif spin_isomer == "para":
		njm = JeM
	elif spin_isomer == "ortho":
		njm = JoM
	else:
		raise ValueError("Invalid spin isomer type. Choose from 'spinless', 'para', or 'ortho'.")

	# Optionally print the calculations if io_write is True
	print("\n**")
	print(colored("Number of basis functions", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("Total |JM> basis functions:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(str(JM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Even J basis functions (JeM):".ljust(LABEL_WIDTH), INFO_COLOR) + colored(str(JeM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Odd J basis functions (JoM):".ljust(LABEL_WIDTH), INFO_COLOR) + colored(str(JoM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"Number of basis functions for {spin_isomer} isomer:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(str(njm).ljust(VALUE_WIDTH), VALUE_COLOR))

	return {
		"JM": JM,
		"JeM": JeM,
		"JoM": JoM,
		"njm": njm
	}


def main():
	# Parse command-line arguments
	args = parse_arguments()
	strength = args.strength
	jmax = args.jmax
	spin_isomer = args.spin

	# No. of grid points along theta and phi
	size_theta = int(2*jmax + 5)
	size_phi = int(2 * (2*jmax + 5))

	# Tolerance limit for a harmitian matrix
	tol = 10e-8

	# print the normalization
	io_write = False
	norm_check = False
	pot_write = False
	debugging = True

	Bconst = 60.853  # cm-1 Taken from NIST data https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000
	CMRECIP2KL = 1.4387672	   	# cm^-1 to Kelvin conversion factor
	Bconst = Bconst * CMRECIP2KL

	# Display input parameters
	display_parameters(strength, jmax, spin_isomer, size_theta, size_phi)

	# Example usage
	file_name = generate_filename(spin_isomer, jmax, strength, size_theta, size_phi, prefix="")
	#print("Generated filename:", file_name)

	xGL, wGL, phixiGridPts, dphixi = compute_legendre_quadrature(size_theta, size_phi, io_write)

	#njm, JM, JeM, JoM = compute_basis_functions(jmax, spin_isomer)
	basis_functions_info = get_number_of_basis_functions_by_spin_isomers(jmax, spin_isomer)


	# Generate (J, M) matrices for each nuclear spin isomer type
	num_basis_functions = basis_functions_info["njm"]
	max_angular_quantum_number = jmax
	spin_isomer_type = spin_isomer
	quantum_numbers_updated = bfunc.generate_linear_rotor_quantum_numbers(num_basis_functions, max_angular_quantum_number, spin_isomer_type)
	#print(quantum_numbers_updated)

	# Real spherical harmonics < cos(theta), phi | JM>
	# basisfun is a 2-dim matrix (size_theta*size_phi, njm)
	basisfun = bfunc.spherical_harmonicsReal(num_basis_functions, size_theta, size_phi, quantum_numbers_updated, xGL, wGL, phixiGridPts, dphixi)

	if (norm_check):
		# Dimension of normMat is (num_basis_functions, num_basis_functions)
		normMat = np.tensordot( basisfun, np.conjugate(basisfun), axes=([0], [0]))
		# Below the function checks normalization condition
		# <lm|l'm'>=delta_ll'mm'
		bfunc.normalization_checkLinear(prefile, strFile, basis_type, basisfun, normMat, num_basis_functions, quantum_numbers_updated, tol)

		basisfun1 = bfunc.spherical_harmonicsComp(num_basis_functions, size_theta, size_phi, quantum_numbers_updated, xGL, wGL, phixiGridPts, dphixi)
		normMat1 = np.tensordot(basisfun, basisfun1, axes=([0], [0]))
		print(normMat1[0])
	whoami()
"""

	# Computation of rotational kinetic energy operator in (lm) basis: H(lm,lm)
		Hrot1 = np.zeros((njm, njm), float)
		for jm in range(njm):
			for jmp in range(njm):
				sum = 0.0
				for s in range(njm):
					sum += np.real(normMat1[jm,
											s] * np.conjugate(normMat1[jmp,
																	   s])) * Bconst * quantum_numbers_updated[s,
																										 0] * (quantum_numbers_updated[s,
																																 0] + 1.0)
				Hrot1[jm, jmp] = sum

	# Computation of potential energy operator in (lm) basis: H(lm,lm)
		v1d = np.zeros((size_theta * size_phi), float)
		for th in range(size_theta):
			for ph in range(size_phi):
				v1d[ph + th * size_phi] = -strength * xGL[th]  # A*cos(theta)

		tempa = v1d[:, np.newaxis] * basisfun
		Hpot = np.tensordot(np.conjugate(basisfun), tempa, axes=([0], [0]))

		if (pot_write):
			bfunc.normalization_checkLinear(
				prefile,
				strFile,
				basis_type,
				v1d,
				Hpot,
				njm,
				quantum_numbers_updated,
				tol)

		Hrot = np.zeros((njm, njm), float)
		for jm in range(njm):
			for jmp in range(njm):
				if (jm == jmp):
					Hrot[jm, jm] = Bconst * quantum_numbers_updated[jm, 0] * \
						(quantum_numbers_updated[jm, 0] + 1.0)

		Htot = Hrot1 + Hpot
		if (np.all(np.abs(Htot - Htot.T) < tol) == False):
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
		reduced_density = np.zeros((njkm, Jmax + 1), dtype=complex)
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
		for th in range(size_theta):
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
