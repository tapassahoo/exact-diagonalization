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
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from sympy.physics.wigner import wigner_3j
from sympy import Rational
from pkg_utils.utils import whoami
from pkg_utils.config import *

from monomer_linear_rotor.basis import (
	generate_monomer_linear_rotor_quantum_numbers,
	count_basis_functions
)
from monomer_linear_rotor.dipole import precompute_monomer_linear_rotor_dipole_terms
from monomer_linear_rotor.hamiltonian import (
	build_monomer_linear_rotor_hamiltonian,
	plot_sparsity,
	diagonalize
)
from monomer_linear_rotor.solver import (
	compute_eigensystem
)
from monomer_linear_rotor.utils import (
	is_hermitian,
	show_simulation_details,
	generate_filename
)	
from monomer_linear_rotor.debug import (
	debug_eigenvalues_eigenvectors
)
from monomer_linear_rotor.io_netcdf import (
	save_all_quantum_data_to_netcdf
)

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

	# Tolerance limit for a harmitian matrix
	deviation_tolerance_value   = 10e-12

	# print the normalization
	compute_rigid_rotor_energy  = False
	orthonormality_check		= False
	hermiticity_check			= True
	#
	display_data				= False

	# Display input parameters
	show_simulation_details(
		potential_strength_cm_inv=potential_strength_cm_inv,   # float, in cm⁻¹
		max_angular_momentum_quantum_number=args.max_angular_momentum_quantum_number,
		spin_state=args.spin,
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

	base_file_name = generate_filename(spin_state, max_angular_momentum_quantum_number, potential_strength_K, args.dipole_moment, args.electric_field, prefix="_")
	
	# All quantum numbers: (J, M)
	all_quantum_numbers = generate_monomer_linear_rotor_quantum_numbers(max_angular_momentum_quantum_number, "spinless")
	# Spin-state-specific quantum numbers
	quantum_numbers_for_spin_state = generate_monomer_linear_rotor_quantum_numbers(max_angular_momentum_quantum_number, spin_state)

	basis_functions_info = count_basis_functions(max_angular_momentum_quantum_number, spin_state)
	total_number_of_states = basis_functions_info["JM"]
	total_number_of_spin_states = basis_functions_info["JM_spin_specific"]
	
	#H_rot = T_rot_einsum + V_rot_einsum
	dipole_terms = precompute_monomer_linear_rotor_dipole_terms(quantum_numbers_for_spin_state, potential_strength_K)
	H_rot = build_monomer_linear_rotor_hamiltonian(quantum_numbers_for_spin_state, B_const_K, dipole_terms)

	# Check Hermiticity
	if hermiticity_check:
		print(colored("\n[INFO] Checking Hermiticity...", "cyan", attrs=["bold"]))
		if is_hermitian(H_rot):
			print(colored("[INFO] Hamiltonian is Hermitian.", "green"))
		else:
			print(colored("[WARNING] Hamiltonian is NOT Hermitian!", "red", attrs=["bold"]))

		# Ensure output directories exist
		plots_dir = os.path.join(args.output_dir, "plots")
		os.makedirs(plots_dir, exist_ok=True)

		# Plot sparsity pattern and save
		sparsity_plot_path = os.path.join(plots_dir, "sparsity_hamiltonian.pdf")
		plot_sparsity(
			H_rot,
			quantum_numbers_for_spin_state,
			save_path=sparsity_plot_path,
			dpi=300,
			max_labels=30,
			color='navy'
		)
		print(colored(f"[INFO] Sparsity plot saved to: {sparsity_plot_path}", "blue"))

	# Diagonalize
	print(f"\nLowest energy eigenvalues for spin type '{spin_state}':")

	# Compute only eigenvalues
	eigenvalues, _, _ = compute_eigensystem(H_rot, num_eig=6, return_vectors=False)

	# Compute and scale
	eigenvalues, eigenvectors, scaled = compute_eigensystem(H_rot, scale_factor=20.0)

	for i, val in enumerate(eigenvalues):
		print(f"  Level {i}: {val:.6f}")

	# Debugging function call
	debug_eigenvalues_eigenvectors(H_rot, eigenvalues, eigenvectors)

	# Output file name
	file_name_netcdf = os.path.join(args.output_dir, "data", f"quantum_data" + base_file_name + ".nc")

	# Prepare arguments
	kwargs = {
		"file_name": file_name_netcdf,
		"cm_inv_to_K": cm_inv_to_K,
		"potential_strength_cm_inv": potential_strength_cm_inv,
		"max_angular_momentum_quantum_number": max_angular_momentum_quantum_number,
		"B_const_cm_inv": B_const_cm_inv,
		"spin_state": spin_state,
		"all_quantum_numbers": all_quantum_numbers,
		"quantum_numbers_for_spin_state": quantum_numbers_for_spin_state,
		"eigenvalues": eigenvalues,
		"eigenvectors": eigenvectors,
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
