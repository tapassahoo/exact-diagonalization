# inspect_file.py

from monomer_linear_rotor.io_netcdf import (
	read_all_attributes,
	inspect_variable,
)

if __name__ == "__main__":
	netcdf_path = "output/data/quantum_data_HCl_spinless_isomer_lmax_20_dipole_moment_1.00D_electric_field_200.00kVcm.nc"
	read_all_attributes(netcdf_path)  # Default prints both global and variable-wise metadata
	inspect_variable(netcdf_path, "eigenvalues", show_data=True, show_plot=False, slice_index=2)


