# inspect_file.py
import os
import numpy as np

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
	quantum_data_root_dir="/Volumes/Schrodinger/pcsa-backup/outputs-of-exeact-diagonalization/"
	if False:
		#netcdf_path = "output/spinless_HF_jmax_4_field_100.00kV_per_cm/data/quantum_data_spinless_HF_jmax_4_field_100.00kV_per_cm.nc"
		netcdf_path = os.path.join(quantum_data_root_dir, f"spinless_HF_jmax_20_field_100.00kV_per_cm/data/quantum_data_spinless_HF_jmax_20_field_100.00kV_per_cm.nc")
		read_all_attributes(netcdf_path)  # Default prints both global and variable-wise metadata
		inspect_variable(netcdf_path, "eigenvalues", show_data=True, show_plot=False, slice_index=2)
		whoami()
		
	jmax_list = np.array([20, 30, 40], dtype=int)

	file_template = "spinless_HF_jmax_{jmax}_field_100.00kV_per_cm/data/quantum_data_spinless_HF_jmax_{jmax}_field_100.00kV_per_cm.nc"

	plot_eigenvalue_convergence(
		jmax_list,
		os.path.join(quantum_data_root_dir, file_template),
		num_levels_to_show=5
	)

	plot_eigenvalue_convergence(
		jmax_list,
		os.path.join(quantum_data_root_dir, file_template),
		level_wanted=1
	)
