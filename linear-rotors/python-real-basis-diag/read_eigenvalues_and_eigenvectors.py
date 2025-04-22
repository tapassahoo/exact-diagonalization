import numpy as np
from netCDF4 import Dataset

def inspect_netcdf_file(filename):
	"""Inspect a NetCDF file and list its contents."""
	with Dataset(filename, "r") as ncfile:
		# Display file metadata
		print("File Metadata:")
		print(f"Title: {ncfile.title}")
		print(f"Description: {ncfile.description}")
		print(f"History: {ncfile.history}")
		print(f"Source: {ncfile.source}")

		# List all available dimensions
		print("\nDimensions:")
		for dim_name, dim in ncfile.dimensions.items():
			print(f"{dim_name}: {len(dim)}")

		# List all available variables and their attributes
		print("\nVariables:")
		for var_name, var in ncfile.variables.items():
			print(f"Variable: {var_name}")
			print(f"  Dimensions: {var.dimensions}")
			print(f"  Shape: {var.shape}")
			print(f"  Data Type: {var.dtype}")
			print(f"  Units: {getattr(var, 'units', 'N/A')}")
			print(f"  Long Name: {getattr(var, 'long_name', 'N/A')}")
			
			# Print attributes for each variable
			print(f"  Attributes:")
			for attr in var.ncattrs():
				attr_value = getattr(var, attr)
				print(f"	{attr}: {attr_value}")


def read_quantum_data(filename, spin_state_name):
    """Read quantum numbers, eigenvalues, and eigenvectors from a NetCDF file."""
    with Dataset(filename, "r") as ncfile:
        # Read quantum numbers and eigen data
        all_quantum_numbers = ncfile.variables["all_quantum_numbers"][:]
        
        # Access the correct variable for spin state quantum numbers
        spin_state_qn_array = ncfile.variables[f"{spin_state_name}_quantum_numbers"][:]
        
        eigenvalues = ncfile.variables["eigenvalues"][:]
        eigenvectors_real = ncfile.variables["eigenvectors_real"][:]
        eigenvectors_imag = ncfile.variables["eigenvectors_imag"][:]
        
        # Reconstruct complex eigenvectors
        eigenvectors = eigenvectors_real + 1j * eigenvectors_imag

    return all_quantum_numbers, spin_state_qn_array, eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors

# Example usage
filename = 'quantum_data.nc'
inspect_netcdf_file(filename)
spin_state_name = 'spinless'
all_quantum_numbers, spin_state_qn_array, eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors = read_quantum_data(filename, spin_state_name)

# Print results (optional)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors (Complex):", eigenvectors)
