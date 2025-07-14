# inspect_file.py

from monomer_linear_rotor.io_netcdf import (
	read_all_attributes,
	inspect_variable,
)
from monomer_linear_rotor.thermo import (
	read_all_quantum_data_files_with_thermo,
)
from monomer_linear_rotor.analysis import (
	plot_eigenvalue_convergence
)

from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


if __name__ == "__main__":
	netcdf_path = "output/spinless_HCl_jmax_10_field_200.00kV_per_cm/data/quantum_data_HCl_spinless_isomer_lmax_10_dipole_moment_1.83D_electric_field_200.00kVcm.nc"
	netcdf_path = "output/spinless_HCl_jmax_20_field_100.00kV_per_cm/data/quantum_data_HCl_spinless_isomer_lmax_20_dipole_moment_1.03D_electric_field_100.00kVcm.nc"
	netcdf_path = "output/spinless_HBr_jmax_20_field_100.00kV_per_cm/data/quantum_data_HBr_spinless_isomer_lmax_20_dipole_moment_0.78D_electric_field_100.00kVcm.nc"
	read_all_attributes(netcdf_path)  # Default prints both global and variable-wise metadata
	inspect_variable(netcdf_path, "eigenvalues", show_data=True, show_plot=False, slice_index=2)

	jmax_list = list(range(10, 21, 2))
	file_template = "output/spinless_HF_jmax_{jmax}_field_200.00kV_per_cm/data/quantum_data_HF_spinless_isomer_lmax_{jmax}_dipole_moment_1.83D_electric_field_200.00kVcm.nc"

	# To plot and annotate multiple levels:
	plot_eigenvalue_convergence(jmax_list, file_template, num_levels_to_show=8)

	# To focus only on level 3:
	plot_eigenvalue_convergence(jmax_list, file_template, level_wanted=1)


	"""
	read_all_quantum_data_files_with_thermo(
		base_output_dir="/Users/tapas/academic-project/outputs/output_spinless_HF_monomer_in_field",
		dipole_moment_D=1.83,
		electric_field_kVcm_list=[0.1] + list(range(10, 201, 10)),
		max_angular_momentum_list=list(range(10, 11, 5)),
		#temperature_list=list(range(2, 50, 2))+list(range(50, 201, 5)),
		temperature_list=list(range(10, 51, 5)),
		spin_type="spinless",
		unit_want="J/mol",
		export_csv=True,
		export_plot=True,
		output_summary_dir="/Users/tapas/academic-project/results/result_spinless_HF_monomer_in_field"
	)
	"""

