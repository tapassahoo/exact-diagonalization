# main.py

import argparse
import os
import inspect
import sys
from pathlib import Path
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

from monomer_linear_rotor.cli import (
	parse_arguments,
	show_dry_run_summary
)
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
	rephase_eigenvectors_real,
	analyze_matrix,
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

def main():
	# --- Parse user input from CLI ---
	args = parse_arguments()

	# --- Dry run: show configuration and exit ---
	if args.dry_run:
		show_dry_run_summary(args)
		sys.exit(0)

	# Execution of main simulation logic goes here (to be implemented)
	B_const_cm_inv = args.B_const
	potential_strength_cm_inv = args.potential_strength	

	# --- Construct output directory structure ---
	os.makedirs(args.output_dir, exist_ok=True)

	# Use the molecule and spin arguments as-is, preserving original casing
	molecule_name = args.molecule if args.molecule else "Unknown"
	spin_label = args.spin if args.spin else "spinless"
	subdir_name = f"{args.spin}_{args.molecule}_jmax_{args.max_angular_momentum_quantum_number}_field_{args.electric_field:.2f}kV_per_cm"

	# Final output path
	output_root_dir = os.path.join(args.output_dir, subdir_name)
	os.makedirs(output_root_dir, exist_ok=True)

	max_angular_momentum_quantum_number = args.max_angular_momentum_quantum_number
	spin_state					= args.spin

	# print the normalization
	compute_rigid_rotor_energy  = False
	hermiticity_check			= True

	# Display input parameters
	show_simulation_details(
		output_root_dir=output_root_dir,
		B_const_cm_inv=B_const_cm_inv,
		potential_strength_cm_inv=potential_strength_cm_inv, # float, in cm⁻¹
		max_angular_momentum_quantum_number=args.max_angular_momentum_quantum_number,
		spin_state=args.spin,
		dipole_moment_D=args.dipole_moment,					 # float or None
		electric_field_kVcm=args.electric_field,			 # float or None
		computed_muE_cm_inv=potential_strength_cm_inv if args.potential_strength is None else None
	)

	if compute_rigid_rotor_energy:
		energies = rotational_energy_levels(B_const_cm_inv, 10)
		plot_rotational_levels(energies)

	base_file_name = generate_filename(molecule_name, spin_state, max_angular_momentum_quantum_number, potential_strength_cm_inv, args.dipole_moment, args.electric_field, prefix=f"")
	
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
	result = analyze_matrix(H_rot)

	print()
	print(colored("Quantum Mechanical Operator Diagnostics:", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("[ ] H_rot matrix is real):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{result['is_real']}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] H_rot is symmetric (H = H.T):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{result['is_symmetric']}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] H_rot is hermitian (H = H†):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{result['is_hermitian']}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Eigenvectors of H_rot matrix are real-valued):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{result['eigenvectors_real']}".ljust(VALUE_WIDTH), VALUE_COLOR))
	#print(result["eigenvectors"][-2,:])
	whoami()

	# Check Hermiticity
	if hermiticity_check:
		print(colored("\n[INFO] Checking Hermiticity...", "cyan", attrs=["bold"]))
		if is_hermitian(H_rot):
			print(colored("[INFO] Hamiltonian is Hermitian.", "green"))
		else:
			print(colored("[WARNING] Hamiltonian is NOT Hermitian!", "red", attrs=["bold"]))

		# Ensure output directories exist
		plots_dir = os.path.join(output_root_dir, "plots")
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
		print(colored("[INFO] ", INFO_COLOR) + 
			  colored("Sparsity plot saved to: ", LABEL_COLOR) + 
			  colored(sparsity_plot_path, VALUE_COLOR))

	# Diagonalize
	check_residual=False
	eigenvalues, eigenvectors = compute_eigensystem(H_rot, check_residual)
	display_eigenvalues(eigenvalues, spin_state)
	# Debugging function call
	debug_eigenvalues_eigenvectors(H_rot, eigenvalues, eigenvectors)

	# Output file name
	# First, create the directory
	output_data_dir = os.path.join(output_root_dir, "data")
	os.makedirs(output_data_dir, exist_ok=True)

	# Then, build the NetCDF filename
	file_name_netcdf = os.path.join(output_data_dir, f"quantum_data{base_file_name}.nc")
	print(colored("[INFO] ", INFO_COLOR) + colored("All quantum data will be saved to: ", LABEL_COLOR) + colored(file_name_netcdf, VALUE_COLOR))


	# Prepare arguments
	kwargs = {
		"file_name": file_name_netcdf,
		"molecule_name": molecule_name,
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

	path = Path(file_name_netcdf)

	if path.exists():
		print(f"[INFO] File exists: {file_name_netcdf}")
	else:
		print(f"[WARNING] File does not exist: {file_name_netcdf}")

	print("\n\nHURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES")

if __name__ == "__main__":
	main()
