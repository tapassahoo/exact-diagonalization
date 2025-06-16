import numpy as np
from netCDF4 import Dataset
import thermodynamics_kelvin as tk


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

if False:
	def read_all_attributes(filename):
		with Dataset(filename, 'r') as ncfile:
			print("=== Global Attributes ===")
			for attr in ncfile.ncattrs():
				print(f"{attr}: {ncfile.getncattr(attr)}")

			print("\n=== Variable-wise Attributes ===")
			for var_name in ncfile.variables:
				var = ncfile.variables[var_name]
				print(f"\nVariable: {var_name}")
				for attr in var.ncattrs():
					print(f"  {attr}: {var.getncattr(attr)}")

def read_all_attributes(filename):
	with Dataset(filename, 'r') as ncfile:
		print("\n📌 Global Attributes")
		print("-" * 60)
		if ncfile.ncattrs():
			for attr in ncfile.ncattrs():
				print(f"{attr:30}: {ncfile.getncattr(attr)}")
		else:
			print("No global attributes found.")

		print("\n📦 Variable-wise Attributes")
		print("-" * 60)
		for var_name in ncfile.variables:
			var = ncfile.variables[var_name]
			print(f"\n🔹 Variable: {var_name}")
			print("-" * (11 + len(var_name)))
			if var.ncattrs():
				for attr in var.ncattrs():
					print(f"  {attr:28}: {var.getncattr(attr)}")
			else:
				print("  No attributes found.")
	print("\n**")

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
filename = "output/quantum_data_for_H2_spinless_isomer_max_angular_momentum_quantum_number4_potential_strength4.471718533716K_grids_theta13_phi31.nc"
read_all_attributes(filename)
all_variables = inspect_netcdf_file(filename)
scalar_data = read_scalar_like_parameters_with_units(filename, all_variables)
display_scalar_parameters(scalar_data)

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
