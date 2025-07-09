import os
import inspect
import sys
import getpass
import socket
import platform
import numpy as np
from datetime import datetime
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
import io
from pkg_utils.utils import whoami
from pkg_utils.config import *

def save_all_quantum_data_to_netcdf(
	file_name,
	cm_inv_to_K,
	potential_strength_cm_inv,
	max_angular_momentum_quantum_number,
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
	- real_eigenvectors (ndarray): Real parts of eigenvectors (shape Nxk or kxN).
	- imag_eigenvectors (ndarray): Imaginary parts of eigenvectors (same shape as real).
	"""

	eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
	real_eigenvectors = np.asarray(real_eigenvectors, dtype=np.float64)
	imag_eigenvectors = np.asarray(imag_eigenvectors, dtype=np.float64)

	k = eigenvalues.shape[0]

	# Ensure eigenvectors are of shape (k, N): k eigenvectors, each of length N
	if real_eigenvectors.shape[0] != k:
		if real_eigenvectors.shape[1] == k:
			real_eigenvectors = real_eigenvectors.T
			imag_eigenvectors = imag_eigenvectors.T
		else:
			raise ValueError(f"Shape mismatch: eigenvectors shape {real_eigenvectors.shape} "
							 f"does not match eigenvalues length {k}.")

	state_count = k
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

