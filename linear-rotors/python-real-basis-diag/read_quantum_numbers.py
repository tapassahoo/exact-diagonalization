import pandas as pd
from netCDF4 import Dataset

def read_and_display_netcdf_quantum_numbers(netcdf_filename):
	"""
	Reads a NetCDF file containing quantum numbers and displays them in tabular format.

	Parameters:
		netcdf_filename (str): Path to the NetCDF file.
	"""

	with Dataset(netcdf_filename, "r") as ncfile:
		print("\n📄 File Description:")
		print(f"  └─ {ncfile.description}")
		print(f"  └─ Created by: {ncfile.history}")
		print(f"  └─ Source: {ncfile.source}")

		# Read variable names
		var_names = list(ncfile.variables.keys())
		print("\n🔍 Variables Found:", var_names)

		# Loop over each variable and display as table
		for var_name in var_names:
			data = ncfile.variables[var_name][:]
			df = pd.DataFrame(data, columns=["J", "M"])
			print(f"\n📘 Quantum Numbers from variable: '{var_name}'")
			print(df.to_string(index=False))


netcdf_file = "quantum_numbers_spinless_isomer_20250420_115426.nc"
read_and_display_netcdf_quantum_numbers(netcdf_file)

