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
	if False: 
		netcdf_path = "/Users/tapas/academic-project/outputs/output/spinless_HF_jmax_40_field_100.00kV_per_cm/data/quantum_data_spinless_HF_jmax_40_field_100.00kV_per_cm.nc"
		read_all_attributes(netcdf_path)  # Default prints both global and variable-wise metadata
		inspect_variable(netcdf_path, "eigenvalues", show_data=True, show_plot=False, slice_index=2)

	jmax_list = list(range(20, 41, 5))
	file_template = "/Users/tapas/academic-project/outputs/output/spinless_HF_jmax_{jmax}_field_100.00kV_per_cm/data/quantum_data_spinless_HF_jmax_{jmax}_field_100.00kV_per_cm.nc"

	# To plot and annotate multiple levels:
	plot_eigenvalue_convergence(jmax_list, file_template, num_levels_to_show=5)

	# To focus only on level 3:
	plot_eigenvalue_convergence(jmax_list, file_template, level_wanted=1)
