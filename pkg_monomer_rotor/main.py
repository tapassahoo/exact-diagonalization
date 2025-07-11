# main.py

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
	rotational_energy_levels,
	plot_rotational_levels,
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
	generate_filename,
	display_eigenvalues,
	convert_dipole_field_energy_to_cm_inv  # or appropriate import
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
	of a linear quantum rotor in an external orienting potential.

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
			"Computation of eigenvalues and eigenfunctions by exact diagonalization of the analytical Hamiltonian for a polar, rigid, linear rotor in an external electric field. The dipole–field interaction is treated explicitly, and the potential energy matrix elements are evaluated using Wigner 3-j symbols."
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
			args.potential_strength = convert_dipole_field_energy_to_cm_inv(args.dipole_moment, args.electric_field) 
		else:
			print("Error: You must provide either --potential-strength or both --dipole-moment and --electric-field.")
			sys.exit(1)

	return args

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
	elif args.dipole_moment is not None and args.electric_field is not None:
		potential_strength_cm_inv = convert_dipole_field_energy_to_cm_inv(args.dipole_moment * args.electric_field)
		print(f"Computed potential strength (mu*E) = {potential_strength_cm_inv:.4f} cm⁻¹")
	else:
		raise ValueError(
			"Missing potential parameters: provide either --potential-strength "
			"or both --dipole-moment-D and --electric-field-kVcm."
		)

	max_angular_momentum_quantum_number = args.max_angular_momentum_quantum_number
	spin_state					= args.spin

	# Spectroscopic constant (B) in cm⁻¹ taken from NIST data
	#B_const_cm_inv = 20.95373 # HF
	B_const_cm_inv = 10.44 # https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=josa&uri=josa-52-1-1&html=true

	# print the normalization
	compute_rigid_rotor_energy  = False
	hermiticity_check			= True

	# Display input parameters
	show_simulation_details(
		B_const_cm_inv=B_const_cm_inv,
		potential_strength_cm_inv=potential_strength_cm_inv,   # float, in cm⁻¹
		max_angular_momentum_quantum_number=args.max_angular_momentum_quantum_number,
		spin_state=args.spin,
		dipole_moment_D=args.dipole_moment,					 # float or None
		electric_field_kVcm=args.electric_field,			 # float or None
		computed_muE_cm_inv=potential_strength_cm_inv if args.potential_strength is None else None
	)

	if compute_rigid_rotor_energy:
		energies = rotational_energy_levels(B_const_cm_inv, 10)
		plot_rotational_levels(energies)

	base_file_name = generate_filename(spin_state, max_angular_momentum_quantum_number, potential_strength_cm_inv, args.dipole_moment, args.electric_field, prefix="_")
	
	# All quantum numbers: (J, M)
	all_quantum_numbers = generate_monomer_linear_rotor_quantum_numbers(max_angular_momentum_quantum_number, "spinless")
	# Spin-state-specific quantum numbers
	quantum_numbers_for_spin_state = generate_monomer_linear_rotor_quantum_numbers(max_angular_momentum_quantum_number, spin_state)

	basis_functions_info = count_basis_functions(max_angular_momentum_quantum_number, spin_state)
	total_number_of_states = basis_functions_info["JM"]
	total_number_of_spin_states = basis_functions_info["JM_spin_specific"]
	
	#H_rot = T_rot_einsum + V_rot_einsum
	dipole_terms = precompute_monomer_linear_rotor_dipole_terms(quantum_numbers_for_spin_state, potential_strength_cm_inv)
	H_rot = build_monomer_linear_rotor_hamiltonian(quantum_numbers_for_spin_state, B_const_cm_inv, dipole_terms)

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
	print(f"\n[ ] Lowest energy eigenvalues for spin type '{spin_state}':")

	# Compute and scale
	eigenvalues, eigenvectors = compute_eigensystem(H_rot)
	display_eigenvalues(eigenvalues)
	# Debugging function call
	debug_eigenvalues_eigenvectors(H_rot, eigenvalues, eigenvectors)

	# Output file name
	# First, create the directory
	output_data_dir = os.path.join(args.output_dir, "data")
	os.makedirs(output_data_dir, exist_ok=True)

	# Then, build the NetCDF filename
	file_name_netcdf = os.path.join(output_data_dir, f"quantum_data{base_file_name}.nc")


	# Prepare arguments
	kwargs = {
		"file_name": file_name_netcdf,
		"max_angular_momentum_quantum_number": max_angular_momentum_quantum_number,
		"spin_state": spin_state,
		"B_const_cm_inv": B_const_cm_inv,
		"potential_strength_cm_inv": potential_strength_cm_inv,
		"all_quantum_numbers": all_quantum_numbers,
		"quantum_numbers_for_spin_state": quantum_numbers_for_spin_state,
		"eigenvalues": eigenvalues,
		"eigenvectors": eigenvectors
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
