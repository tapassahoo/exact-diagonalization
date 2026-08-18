import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
from pathlib import Path

from monomer_linear_rotor.thermo import (
	read_all_quantum_data_files_with_thermo,
	get_ground_state_dipole_orientation,
	plot_cv_comparison,
	plot_dipole_panel,
	plot_dipole_orientation_comparison,
	compute_angular_probability_density,
	#plot_dipole_orientation_3d,
	#plot_all_molecules_3d,
)

from pkg_utils.utils import whoami
from pkg_utils.env_report import whom

def get_temperature_list(grid="default"):
	"""
	Return a predefined temperature grid.

	Parameters
	----------
	grid : {"default", "orientation", "heat_capacity"}
		Name of the temperature grid to return.

	Returns
	-------
	numpy.ndarray
		Temperature values in K.

	Raises
	------
	ValueError
		If an unsupported grid name is provided.
	"""

	base_grid = np.arange(0.1, 100.1, 0.1)

	if grid == "orientation":
		return np.concatenate([
			[0.01],
			np.arange(2.0, 101.0, 1.0),
			[200.0],
		])

	if grid in {"default", "heat_capacity"}:
		return base_grid

	raise ValueError(
		f"Unknown grid '{grid}'. "
		"Choose from 'default', 'orientation', or 'heat_capacity'."
	)

#temperature_list = get_temperature_list("heat_capacity")
#print("Temperature list:")
#print([f"{T:.2f}" for T in temperature_list])

quantum_data_root_dir="/Volumes/Schrodinger/pcsa-backup/outputs-of-exeact-diagonalization/"
#jmax_list=list(range(20, 41, 5))
jmax_list=[60]
#electric_field_list=[100, 200, 300, 400]
electric_field_list=[400]
unit_want="wavenumber"
#unit_want="SI",

all_results = {}
for mol in ["HI"]:
#for mol in ["HF", "HCl", "HBr", "HI"]:
	thermo_dict = read_all_quantum_data_files_with_thermo(
		quantum_data_root_dir=quantum_data_root_dir,
		molecule=mol,
		electric_field_list=electric_field_list,
		jmax_list=jmax_list,
		#temperature_list=get_temperature_list("heat_capacity"),
		temperature_list=get_temperature_list(),
		spin_type="spinless",
		unit_want=unit_want,
		export_csv=False,
		export_plot=False,
		output_summary_dir="/Users/tapas/academic-project/results/"
	)
	all_results[mol] = thermo_dict


if False:
	get_ground_state_dipole_orientation(
		all_results,
		get_temperature_list,
	)

	plot_dipole_orientation_3d(
		thermo_dict_by_molecule=all_results,
		get_temperature_list=get_temperature_list,
		out_dir="/Users/tapas/academic-project/results/dipole_orientation_3D"
	)

	plot_all_molecules_3d(
		thermo_dict_by_molecule=all_results,
		get_temperature_list=get_temperature_list,
		out_path="/Users/tapas/academic-project/results/dipole_orientation/all_molecules_3D.png",
	)

	plot_dipole_orientation_comparison(
		thermo_dict_by_molecule=all_results,
		get_temperature_list=get_temperature_list,
		unit_want=unit_want,
		out_dir = f"/Users/tapas/academic-project/results/"
	)

	out_dir = Path("/Users/tapas/academic-project/results/")
	out_dir.mkdir(parents=True, exist_ok=True)

	save_path = out_dir / "dipole_orientation_comparison.png"

	plot_dipole_orientation_comparison(
		thermo_dict_by_molecule=all_results,
		electric_field_list=electric_field_list,
		get_temperature_list=get_temperature_list,
		save_path=save_path,
	)

	plt.show()


if False:
	filename = f"Cv_rot_{mol}_E{electric_field_list[0]}kVcm_upto_100K.png"
	#filename = f"Cv_rot_E{electric_field_list[0]}kVcm_upto_100K.png"
	plot_cv_comparison(
		thermo_dict_by_molecule=all_results,
		get_temperature_list=get_temperature_list,
		unit_want=unit_want,
		out_path = f"/Users/tapas/academic-project/results/{filename}"
	)
