import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import logging

from monomer_linear_rotor.thermo import (
	read_all_quantum_data_files_with_thermo,
	plot_cv_comparison,
	plot_dipole_orientation_comparison,
	get_ground_state_dipole_orientation,
	compute_angular_probability_density,
)

from pkg_utils.utils import whoami
from pkg_utils.env_report import whom

if False:
	def get_temperature_list(molecule: str):
		"It is for the computation of heat capacity."
		if molecule == "HF":
			return np.unique(np.concatenate([
				np.linspace(0.5, 5.0, 5),
				np.arange(5.0, 20.5 + 1e-12, 0.25),
				np.arange(20.0, 40.0 + 1e-12, 1.25),
				np.arange(40.0, 202.0 + 1e-12, 1.5),
				np.array([100.0, 200.0])  # explicitly include
			]))
			#return sorted(set(
			#	list(np.arange(0.5, 20.1, 0.25)) +
			#	list(range(20, 41, 1)) +
			#	list(range(40, 101, 1))
			#))

		elif molecule == "HCl":
			return np.unique(np.concatenate([
				np.arange(0.5, 3.0 + 1e-9, 0.5),
				np.arange(3.1, 8.5 + 1e-9, 0.1),
				np.arange(8.2, 10.6 + 1e-9, 0.2),
				np.arange(10.0, 20.0 + 1e-9, 1.0),
				np.arange(20.0, 100.0 + 1e-9, 1.0),
				np.array([100.0, 200.0])  # explicitly include
			]))
			#return sorted(set(
			#	list(np.arange(0.2, 12.1, 0.2)) +
			#	list(np.arange(12.5, 20.1, 0.5)) +
			#	list(np.arange(20.0, 40.1, 1.0)) +
			#	list(range(40, 101, 2))
			#))
		elif molecule == "HBr":
			return np.unique(np.concatenate([
				np.arange(0.2, 2.0 + 1e-9, 0.5),
				np.arange(2.0, 5.2 + 1e-9, 0.05),
				np.arange(5.1, 8.0 + 1e-9, 0.1),
				np.arange(8.0, 10.6 + 1e-9, 0.2),
				np.arange(10.0, 100.0 + 1e-9, 1.0),
				np.array([100.0, 200.0])  # explicitly include
			]))
		elif molecule == "HI":
			return np.unique(np.concatenate([
				np.arange(0.2, 2.0 + 1e-9, 0.5),
				np.arange(2.0, 5.2 + 1e-9, 0.05),
				np.arange(5.1, 8.0 + 1e-9, 0.1),
				np.arange(8.0, 10.6 + 1e-9, 0.2),
				np.arange(10.0, 100.0 + 1e-9, 1.0),
				np.array([100.0, 200.0])  # explicitly include
			]))
			#return sorted(set(
			#	list(np.arange(0.2, 10.1, 0.1)) +
			#	list(np.arange(10.0, 20.1, 0.5)) +
			#	list(np.arange(20.0, 40.1, 1.0)) +
			#	list(range(40, 101, 1))
			#))

		else:
			raise ValueError(f"Unsupported molecule: {molecule}")


def get_temperature_list(
	molecule: str,
	dipole_orientation: bool = False,
	heat_capacity: bool = False
):
	"""
	Returns temperature grid depending on the requested property.

	Parameters:
		molecule (str)
		dipole_orientation (bool): grid optimized for <cos(theta)>
		heat_capacity (bool): grid optimized for Cv
	"""

	if dipole_orientation:
		# Coarse grid (your requirement)
		temps = np.concatenate((
			np.array([0.01]),
			np.arange(2, 101, 1)
		))

	elif heat_capacity:
		# Heat capacity needs smooth derivatives → dense grid
		temps = np.linspace(0.01, 100.0, 1000)

	else:
		# Default (general purpose)
		temps = np.linspace(0.01, 100.0, 500)

	return temps


def plot_cv_surface(
	thermo_dict_by_field,
	temperature_list,
	unit="Kelvin",
	mode="surface",  # "surface" or "wireframe"
	out_path=None,
	fixed_lmax=None,
	title=r"$C_v(T, E)$ Surface"
):
	"""
	Plot a 3D surface or wireframe of Cv vs Temperature and Electric Field.

	Parameters:
		thermo_dict_by_field (dict): {(lmax, E_field): {T: {Cv data}}}
		temperature_list (list): Temperatures in K.
		unit (str): Cv unit: "Kelvin", "J/mol", or "eV".
		mode (str): "surface" or "wireframe"
		out_path (str): File path to save plot; if None, display.
		fixed_lmax (int or None): If specified, restrict to this lmax only.
		title (str): Title of the plot.
	"""

	# Enable LaTeX rendering
	mpl.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"axes.labelsize": 12,
		"font.size": 12,
		"legend.fontsize": 11,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10
	})

	# Extract relevant keys
	selected_items = {
		(l, E): thermo for (l, E), thermo in thermo_dict_by_field.items()
		if fixed_lmax is None or l == fixed_lmax
	}

	if not selected_items:
		print("[!] No matching entries found for plot.")
		return

	E_fields = sorted(set(E for (_, E) in selected_items.keys()))
	T_vals = sorted(temperature_list)

	# Build meshgrid
	T_grid, E_grid = np.meshgrid(T_vals, E_fields)
	Cv_grid = np.full_like(T_grid, fill_value=np.nan, dtype=float)

	# Fill Cv values into grid
	for i, E in enumerate(E_fields):
		for j, T in enumerate(T_vals):
			for (l, E_val), thermo in selected_items.items():
				if E_val == E:
					try:
						Cv_grid[i, j] = thermo[T][f"Cv ({unit}/K)"]
					except KeyError:
						Cv_grid[i, j] = np.nan
					break

	# Begin plotting
	fig = plt.figure(figsize=(8, 5))
	ax = fig.add_subplot(111, projection="3d")

	if mode == "surface":
		surf = ax.plot_surface(T_grid, E_grid, Cv_grid, cmap="viridis", edgecolor="none")
	elif mode == "wireframe":
		surf = ax.plot_wireframe(T_grid, E_grid, Cv_grid, color="gray")

	# Labels
	ax.set_xlabel(r"Temperature $T$ (K)")
	ax.set_ylabel(r"Electric Field $E$ (kV/cm)")
	ax.set_zlabel(rf"Heat Capacity $C_v$ ({unit}/K)")
	ax.set_title(title, fontsize=14)

	# Color bar
	fig.colorbar(surf, shrink=0.6, aspect=12, label=rf"$C_v$ ({unit}/K)")

	plt.tight_layout()
	if out_path:
		plt.savefig(out_path, dpi=300)
		print(f"[✓] 3D plot saved to: {out_path}")
	else:
		plt.show()
	plt.close()


def plot_cv_overlay(thermo_dict_by_field, unit="Kelvin", out_path=None):
	"""
	Overlay Cv vs T curves for different electric fields.

	Parameters:
		thermo_dict_by_field (dict): {E_field: thermo_data_dict}
		unit (str): Cv unit
		out_path (str): Output image path
	"""
	plt.figure(figsize=(7, 5))

	for E in sorted(thermo_dict_by_field):
		thermo = thermo_dict_by_field[E]
		T_vals = sorted(thermo.keys())
		Cv_vals = [thermo[T][f"Cv ({unit}/K)"] for T in T_vals]
		plt.plot(T_vals, Cv_vals, label=f"{E:.2f} kV/cm")

	plt.xlabel("Temperature (K)")
	plt.ylabel(f"Heat Capacity ({unit}/K)")
	plt.title("Heat Capacity vs Temperature for Different Fields")
	plt.grid(True)
	plt.legend(title="Electric Field")
	plt.tight_layout()
	if out_path:
		plt.savefig(out_path, dpi=300)
		print(f"[✓] Cv overlay plot saved to: {out_path}")
	else:
		plt.show()
	plt.close()

def plot_cv_heatmap(
	thermo_dict_by_field,
	temperature_list,
	unit="Kelvin",
	out_path=None,
	fixed_lmax=None,
	title=r"$C_v(T, E)$ Heatmap",
	export_txt=False,
	txt_path=None
):
	"""
	Plot a heatmap of Cv vs Temperature and Electric Field for a fixed lmax, with annotations.
	Optionally export the matrix as a .txt or .csv file.

	Parameters:
		thermo_dict_by_field (dict): {(lmax, E_field): {T: {"Cv": ...}}}
		temperature_list (list): Temperatures in K (x-axis).
		unit (str): Unit for Cv ("Kelvin", "J/mol", or "eV").
		out_path (str): Path to save the plot. If None, shows it.
		fixed_lmax (int): Fixed value of lmax to filter from dictionary.
		title (str): Title of the plot.
		export_txt (bool): If True, export data matrix to .txt or .csv.
		txt_path (str): Path to save the exported file.
	"""

	mpl.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"axes.labelsize": 12,
		"font.size": 12
	})

	selected_items = {
		E: thermo for (l, E), thermo in thermo_dict_by_field.items()
		if l == fixed_lmax
	}

	if not selected_items:
		print(f"[!] No data found for lmax = {fixed_lmax}")
		return

	E_fields = sorted(selected_items.keys())
	T_vals = sorted(temperature_list)

	# Fill matrix
	Cv_matrix = np.full((len(E_fields), len(T_vals)), np.nan)
	for i, E in enumerate(E_fields):
		thermo = selected_items[E]
		for j, T in enumerate(T_vals):
			try:
				Cv_matrix[i, j] = thermo[T][f"Cv ({unit}/K)"]
			except KeyError:
				Cv_matrix[i, j] = np.nan

	# Plot heatmap
	fig, ax = plt.subplots(figsize=(8, 5))
	im = ax.imshow(
		Cv_matrix,
		aspect='auto',
		origin='lower',
		cmap='viridis',
		extent=[min(T_vals), max(T_vals), min(E_fields), max(E_fields)],
		interpolation='none'
	)

	# Annotate
	for i, E in enumerate(E_fields):
		for j, T in enumerate(T_vals):
			val = Cv_matrix[i, j]
			if not np.isnan(val):
				ax.text(
					T, E, f"{val:.2f}",
					ha='center', va='center',
					color='white' if val > np.nanmean(Cv_matrix) else 'black',
					fontsize=8
				)

	ax.set_xlabel(r"Temperature $T$ (K)")
	ax.set_ylabel(r"Electric Field $E$ (kV/cm)")
	ax.set_title(title + rf" at $l_\mathrm{{max}} = {fixed_lmax}$")
	fig.colorbar(im, ax=ax, label=rf"$C_v$ ({unit}/K)")
	plt.tight_layout()

	if out_path:
		plt.savefig(out_path, dpi=300)
		print(f"[✓] Heatmap saved to: {out_path}")
	else:
		plt.show()

	plt.close()

	# Export as .txt or .csv
	if export_txt:
		df = pd.DataFrame(Cv_matrix, index=E_fields, columns=T_vals)
		df.index.name = "E_field (kV/cm)"
		df.columns.name = "Temperature (K)"
		if txt_path is None:
			txt_path = f"cv_matrix_lmax{fixed_lmax}.txt"
		df.to_csv(txt_path, sep="\t", float_format="%.6f")
		print(f"[✓] Heat capacity matrix saved to: {txt_path}")


		#plot_cv_surface(
		#	thermo_dict_by_field=thermo_dict_by_field,
		#	temperature_list=temperature_list,
		#	unit=unit_want,
		#	mode="surface",  # or "wireframe"
		#	out_path="cv_surface.png"
		#)

		#plot_cv_surface(
		#	thermo_dict_by_field=thermo_dict_by_field,
		#	temperature_list=temperature_list,
		#	unit="J/mol",
		#	mode="wireframe",
		#	out_path="cv_surface_lmax10.png",
		#	fixed_lmax=10,
		#	title=r"Rotational Heat Capacity $C_v(T, E)$ for $l_{\max} = 10$"
		#)

		#plot_cv_heatmap(
		#	thermo_dict_by_field=thermo_dict_by_field,
		#	temperature_list=temperature_list,
		#	unit="J/mol",
		#	fixed_lmax=10,
		#	out_path="cv_heatmap_lmax10_annotated.png",
		#	export_txt=True,
		#	txt_path="cv_matrix_lmax10.txt"
		#)


		#plot_cv_overlay(
		#	thermo_dict_by_field=thermo_dict_by_field,
		#	unit=unit,
		#	out_path="cv_overlay.png"
		#)

#temperature_list = get_temperature_list("HF", dipole_orientation=True)
#print("Temperature list:")
#print([f"{T:.2f}" for T in temperature_list])

quantum_data_root_dir="/Volumes/Schrodinger/pcsa-backup/outputs-of-exeact-diagonalization/"
#jmax_list=list(range(20, 41, 5))
jmax_list=[60]
#electric_field_list=[100, 200, 300, 400, 500]
electric_field_list=[500]
dipole_orientation = True
unit_want="wavenumber"
#unit_want="SI",

all_results = {}
for mol in ["HF"]:
#for mol in ["HF", "HCl", "HBr", "HI"]:
	thermo_dict = read_all_quantum_data_files_with_thermo(
		quantum_data_root_dir=quantum_data_root_dir,
		molecule=mol,
		electric_field_list=electric_field_list,
		jmax_list=jmax_list,
		temperature_list=get_temperature_list(mol, dipole_orientation = True),
		spin_type="spinless",
		unit_want=unit_want,
		export_csv=False,
		export_plot=False,
		output_summary_dir="/Users/tapas/academic-project/results/"
	)
	all_results[mol] = thermo_dict

whoami()

if False:
	get_ground_state_dipole_orientation(
		all_results,
		get_temperature_list,
	)

#filename = f"dipole_orientation_cos_theta_avg_{mol}_E{electric_field_list[0]}kVcm_upto_100K.png"
filename = f"dipole_orientation_cos_theta_avg_E{electric_field_list[0]}kVcm_upto_100K.png"
plot_dipole_orientation_comparison(
	thermo_dict_by_molecule=all_results,
	get_temperature_list=get_temperature_list,
	unit_want=unit_want,
	out_path = f"/Users/tapas/academic-project/results/{filename}"
)

if False:
	filename = f"Cv_rot_{mol}_E{electric_field_list[0]}kVcm_upto_100K.png"
	#filename = f"Cv_rot_E{electric_field_list[0]}kVcm_upto_100K.png"
	plot_cv_comparison(
		thermo_dict_by_molecule=all_results,
		get_temperature_list=get_temperature_list,
		unit_want=unit_want,
		out_path = f"/Users/tapas/academic-project/results/{filename}"
	)
