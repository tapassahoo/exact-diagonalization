from netCDF4 import Dataset
import numpy as np

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

# Example Usage
eigenvalues, scaled_eigenvalues, eigenvectors = load_eigenvalues_eigenvectors_netcdf("eigen_data.nc")
print(eigenvalues)
print(eigenvectors[:,2].conj().T@eigenvectors[:,2])

