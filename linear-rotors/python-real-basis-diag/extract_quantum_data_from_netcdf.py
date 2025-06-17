import os
from itertools import product
import numpy as np
from netCDF4 import Dataset
import thermodynamics_kelvin as tk
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom

import os
from itertools import product
from netCDF4 import Dataset

def read_all_attributes(filename):
	"""Helper function to read and print global and variable-level attributes from a NetCDF file."""
	with Dataset(filename, 'r') as ncfile:
		print("\nGlobal Attributes")
		print("-" * 60)
		for attr in ncfile.ncattrs():
			print(f"{attr:30}: {ncfile.getncattr(attr)}")

		print("\nVariable-wise Attributes")
		print("-" * 60)
		for var_name, var in ncfile.variables.items():
			print(f"\nVariable: {var_name}")
			for attr in var.ncattrs():
				print(f"  {attr:28}: {var.getncattr(attr)}")
	print("\nAttribute inspection complete.\n")

def read_all_quantum_data_files(
	base_output_dir,
	dipole_moment_D,
	electric_field_kVcm_list,
	max_angular_momentum_list,
	spin_type="spinless"
):
	"""
	Read NetCDF data for various configurations of an HF monomer in an electric field.

	Parameters:
		base_output_dir (str): Base path where output directories are stored.
		dipole_moment_D (float): Dipole moment in Debye.
		electric_field_kVcm_list (list): Electric field strengths in kV/cm.
		max_angular_momentum_list (list): Values of maximum angular momentum quantum number (lmax).
		spin_type (str): Rotor spin type: 'spinless', 'ortho', or 'para'.
	"""
	for lmax, E in product(max_angular_momentum_list, electric_field_kVcm_list):
		# Define grid size based on lmax
		theta_grid_count = 2 * lmax + 5
		phi_grid_count = 2 * theta_grid_count + 5

		# Clean subdirectory and filename strings (avoid dots)
		subdir = (
			f"{spin_type}_HF_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm"
		).replace(".", "_")

		filename = (
			f"quantum_data_HF_{spin_type}_isomer_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm_"
			f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}.nc"
		)

		file_path = os.path.join(base_output_dir, subdir, filename)

		print(f"\nChecking file: {file_path}")
		if os.path.exists(file_path):
			try:
				read_all_attributes(file_path)
				all_variables = inspect_netcdf_file(file_path)
				scalar_data = read_scalar_like_parameters_with_units(file_path, all_variables)
				display_scalar_parameters(scalar_data)
				with Dataset(file_path, 'r') as nc:
					print("Variable Summary")
					print("-" * 60)
					for var in nc.variables:
						shape = nc.variables[var].shape
						dtype = nc.variables[var].dtype
						print(f"  {var:30}: shape = {shape}, dtype = {dtype}")
				print("File read successfully.\n")
			except Exception as e:
				print(f"Error reading file: {e}")
		else:
			print("File does not exist.\n")


def inspect_netcdf_file(filename):
	"""Inspect variables and their shapes in a NetCDF file."""
	with Dataset(filename, 'r') as ncfile:
		print("Variables in the NetCDF file:")
		variable_names = []
		for var_name, var in ncfile.variables.items():
			shape_str = ", ".join(str(s) for s in var.shape)
			print(f"- {var_name} (dimensions: {shape_str})")
			variable_names.append(var_name)

		print("\nGlobal Attributes in the NetCDF file:")
		for attr in ncfile.ncattrs():
			print(f"- {attr}")

		return variable_names

def read_scalar_like_parameters_with_units(filename, params):
	"""
	Read scalar or scalar-like (single-value) parameters and their units.
	"""
	scalar_params = {}
	with Dataset(filename, "r") as ncfile:
		for param in params:
			if param in ncfile.variables:
				var = ncfile.variables[param]
				try:
					value = var[()] if var.ndim == 0 else var[:].item() if var.size == 1 else None
				except Exception as e:
					print(f"Error reading '{param}': {e}")
					continue

				if value is not None:
					unit = var.getncattr("units") if "units" in var.ncattrs() else "unknown"
					scalar_params[param] = {"value": value, "unit": unit}
			else:
				print(f"Warning: '{param}' not found in the file.")
	return scalar_params

def display_scalar_parameters(params):
	"""Neatly display scalar parameters."""
	if not params:
		print("\nNo scalar-like parameters found.")
		return

	print("\nScalar-like Parameters with Units:")
	print("-" * 55)
	print(f"{'Parameter':<40} {'Value':>8}   {'Unit'}")
	print("-" * 55)
	for name, data in params.items():
		print(f"{name:<40} {data['value']:>8}   {data['unit']}")
	print("-" * 55)

def read_quantum_data(filename, spin_state_name):
	"""
	Read quantum numbers, eigenvalues, and eigenvectors from a NetCDF file.
	
	Parameters:
	- filename: str, path to the NetCDF file
	
	Returns:
	- Tuple of numpy arrays: 
		all_quantum_numbers, spin_state_qn_array, 
		eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors (complex)
	"""
	with Dataset(filename, "r") as ncfile:
		# Read all quantum numbers
		all_quantum_numbers = ncfile.variables["all_quantum_numbers"][:]
		
		# Automatically determine spin state from available variables
		spin_state_qn_array = ncfile.variables[f"{spin_state_name}_quantum_numbers"][:]
		
		# Read eigenvalues and eigenvectors
		eigenvalues = ncfile.variables["eigenvalues"][:]
		eigenvectors_real = ncfile.variables["eigenvectors_real"][:]
		eigenvectors_imag = ncfile.variables["eigenvectors_imag"][:]

		# Reconstruct complex eigenvectors
		eigenvectors = eigenvectors_real + 1j * eigenvectors_imag

	return all_quantum_numbers, spin_state_qn_array, eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors

# === Usage ===
read_all_quantum_data_files(
	base_output_dir="/Users/tapas/academic-project/outputs/output_spinless_HF_monomer_in_field",
	dipole_moment_D=1.83,
	electric_field_kVcm_list=[0.1] + list(range(10, 201, 10)),
	max_angular_momentum_list=list(range(10, 51, 5)),
	spin_type="spinless"
)

whom()
whoami()

spin_state_name = "spinless"
all_quantum_numbers, quantum_numbers_for_spin_state, eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors = read_quantum_data(filename, spin_state_name)

print("\n📊 Quantum Data Summary\n" + "-"*40)
print(f"🔹 All Quantum Numbers Shape: {all_quantum_numbers.shape}")
print(f"🔹 Spin State Quantum Numbers Shape: {quantum_numbers_for_spin_state.shape}")
print(f"🔹 Eigenvalues Count: {len(eigenvalues)}")
print(f"🔹 Eigenvector Dimension (Real): {eigenvectors_real.shape}")
print(f"🔹 Eigenvector Dimension (Imag): {eigenvectors_imag.shape}")

print("\n🔸 Sample Eigenvalues:")
print(np.array2string(eigenvalues[:5], precision=4, separator=", "))

print("\n🔸 Sample Complex Eigenvector (First Column):")
print(np.array2string(eigenvectors[:, 0], precision=4, separator=", "))

temperatures = np.linspace(1, 500, 100)  # Temperature range from 1 K to 500 K
fixed_temperature = 300  # Example fixed temperature for probability plot

# Plot Probability vs Basis Index for a fixed temperature
tk.plot_probability_profile(eigenvalues, fixed_temperature, threshold=0.001)

# Plot Average Energy vs Temperature
tk.plot_average_energy_vs_temperature(eigenvalues, temperatures)

# Plot Heat Capacity vs Temperature
tk.plot_heat_capacity_vs_temperature(eigenvalues, temperatures)
