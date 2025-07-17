import os
from itertools import product
import numpy as np
from netCDF4 import Dataset
from typing import Optional, Union
import pandas as pd
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.constants import R, k as k_B, N_A, h, c, e as e_charge
import thermodynamics_kelvin as tk
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


k_B_cm = 0.69503476  # Boltzmann constant in cm⁻¹/K
cm_to_J = h * c * 100
cm_to_eV = cm_to_J / e_charge
k_B_cm = k_B / cm_to_J


from scipy.constants import R as GAS_CONSTANT_J_PER_MOL_K
from scipy.constants import k as BOLTZMANN_J_PER_K

def compute_thermo_from_eigenvalues(eigenvalues, temperature_list, unit):
	"""
	Compute thermodynamic properties (Z, populations, U, Cv) from energy eigenvalues.

	Parameters
	----------
	eigenvalues : array_like
		1D array of energy eigenvalues in wavenumber units (cm⁻¹).

	temperature_list : array_like
		List or array of temperatures (in Kelvin) for which properties are computed.

	unit : {'wavenumber', 'SI'}
		Unit for output values:
			- 'wavenumber' : Energy and Cv in cm⁻¹ and cm⁻¹/K
			- 'SI'		 : Energy in J/mol and Cv in J/mol·K

	Returns
	-------
	dict
		Dictionary keyed by temperature (K), each value containing:
			- temperature_K	   : Temperature in Kelvin
			- unit				: 'wavenumber' or 'SI'
			- display_unit		: e.g., 'cm^-1' or 'J/mol'
			- beta				: Inverse temperature (1/kB·T)
			- partition_function  : Canonical partition function Z
			- populations		 : Normalized Boltzmann populations
			- internal_energy	 : U in chosen unit
			- heat_capacity	   : Cv in chosen unit per K
	"""
	# Ensure 1D array
	energies = np.atleast_1d(np.asarray(eigenvalues, dtype=np.float64)).flatten()
	if energies.ndim != 1:
		raise ValueError("Eigenvalues must be a one-dimensional array.")

	if unit not in {"wavenumber", "SI"}:
		raise ValueError("Invalid unit. Choose either 'wavenumber' or 'SI'.")

	# Boltzmann constant in cm⁻¹/K
	KB_CM1_PER_K = 0.69503476

	results = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"Temperature must be strictly positive. Received: {T} K")

		beta = 1.0 / (KB_CM1_PER_K * T)      # unit: 1/cm⁻¹

		# Numerical stability: shift energies
		E_shifted = energies - np.min(energies)
		boltzmann_weights = np.exp(-beta * E_shifted)
		Z = np.sum(boltzmann_weights)
		populations = boltzmann_weights / Z

		# Use unshifted energies for thermodynamic averages
		E_avg = np.dot(populations, energies)
		E2_avg = np.dot(populations, energies**2)
		Cv = beta**2 * (E2_avg - E_avg**2)

		# Unit conversion
		if unit == "wavenumber":
			U_out = E_avg                    # in cm⁻¹
			Cv_out = Cv                      # in cm⁻¹/K
			display_unit = "cm^-1"
			display_cv_unit = "cm^-1/K"
		else:  # SI
			U_out = E_avg * GAS_CONSTANT_J_PER_MOL_K # in J/mol
			Cv_out = Cv * GAS_CONSTANT_J_PER_MOL_K   # in J/mol·K
			display_unit = "J/mol"
			display_cv_unit = "J/mol·K"

		# Store results
		results[T] = {
			"temperature_K": T,
			"unit": unit,
			"display_unit": display_unit,
			"display_cv_unit": display_cv_unit,
			"partition_function": Z,
			"populations": populations,
			"internal_energy": U_out,
			"heat_capacity": Cv_out
		}

	return results

def plot_cv_vs_temperature(
	thermo_data,
	unit="Kelvin",
	out_path=None,
	title=None,
	context="Rotational"
):
	"""
	Plot Cv vs Temperature using LaTeX-rendered labels with automatic context-based titling.

	Parameters:
		thermo_data (dict): {T: {Cv data}}
		unit (str): Units of heat capacity ("Kelvin", "J/mol", or "eV").
		out_path (str): Path to save the plot. If None, displays instead.
		title (str): Custom title. If None, auto-generated from context.
		context (str): Physical context like "Rotational", "Vibrational", etc.
	"""
	# Enable LaTeX rendering in matplotlib
	mpl.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"axes.labelsize": 12,
		"font.size": 12,
		"legend.fontsize": 11,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10
	})

	# Extract data
	T_vals = sorted(thermo_data.keys())
	Cv_vals = [thermo_data[T][f"Cv ({unit}/K)"] for T in T_vals]

	# Set default title
	if title is None:
		title = rf"{context} Heat Capacity $C_V$ vs Temperature $T$"

	# Create plot
	plt.figure(figsize=(6, 4))
	plt.plot(T_vals, Cv_vals, 'o-', label=rf"$C_V$ ({unit}/K)")
	plt.xlabel(r"Temperature $T$ (K)")
	plt.ylabel(rf"Heat Capacity $C_V$ ({unit}/K)")
	plt.title(title)
	plt.grid(True)
	plt.legend()
	plt.tight_layout()

	# Save or show
	if out_path:
		plt.savefig(out_path, dpi=300)
		print(f"[✓] Plot saved to: {out_path}")
	else:
		plt.show()

	plt.close()


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


def save_thermo_with_Z_and_populations(
	thermo_data,
	temperatures,
	eigenvalues,
	unit="wavenumber",
	txt_path="thermo_summary.txt",
	csv_path="thermo_summary.csv",
	save_populations=False,
	population_dir="populations_txt"
):
	"""
	Save thermodynamic data including Z and optional populations to .txt and .csv.

	Parameters:
		thermo_data (dict): Output from compute_thermo_from_eigenvalues().
		temperatures (list): Temperatures (in Kelvin).
		eigenvalues (array): Energy eigenvalues (in the same unit).
		unit (str): Energy unit.
		txt_path (str): Path to output .txt summary.
		csv_path (str): Path to output .csv summary.
		save_populations (bool): If True, save state-wise populations per T.
		population_dir (str): Output directory for per-T population text files.
	"""

	# -----------------------------------
	# Save formatted TXT summary
	# -----------------------------------
	with open(txt_path, "w") as f:
		f.write(f"{'Temperature (K)':>15}  {f'U ({unit})':>15}  {f'Cv ({unit}/K)':>15}  {f'Z':>15}\n")
		f.write("-" * 65 + "\n")
		for T in sorted(temperatures):
			Z = thermo_data[T]["partition_function"]
			U = thermo_data[T]['internal_energy']
			Cv = thermo_data[T]['heat_capacity']
			f.write(f"{T:15.1f}  {U:15.6f}  {Cv:15.6f}  {Z:15.6f}\n")
	print(f"[✓] TXT summary saved: {txt_path}")

	# -----------------------------------
	# Save CSV summary
	# -----------------------------------
	df = pd.DataFrame({
		"Temperature (K)": sorted(temperatures),
		f"U ({unit})": [thermo_data[T]['internal_energy'] for T in sorted(temperatures)],
		f"Cv ({unit}/K)": [thermo_data[T]['heat_capacity'] for T in sorted(temperatures)],
		"Partition Function Z": [thermo_data[T]['partition_function'] for T in sorted(temperatures)]
	})
	df.to_csv(csv_path, index=False)
	print(f"[✓] CSV summary saved: {csv_path}")

	# -----------------------------------
	# Save per-temperature population files (optional)
	# -----------------------------------
	if save_populations:
		os.makedirs(population_dir, exist_ok=True)
		eigenvalues = np.array(eigenvalues)

		for T in sorted(temperatures):
			pops = thermo_data[T].get("Populations")
			if pops is None:
				continue

			pop_file = f"populations_T_{T:.1f}K.txt"
			full_path = os.path.join(population_dir, pop_file)
			with open(full_path, "w") as pf:
				pf.write(f"# Population distribution at T = {T:.1f} K\n")
				pf.write(f"# {'Index':>6}  {'Energy (' + unit + ')':>20}  {'P_i':>15}\n")
				pf.write("#" + "-" * 50 + "\n")
				for i, (Ei, Pi) in enumerate(zip(eigenvalues, pops)):
					pf.write(f"{i:6d}  {Ei:20.6f}  {Pi:15.6e}\n")
			print(f"[✓] Populations saved: {full_path}")

def read_all_quantum_data_files_with_thermo(
	base_output_dir: str,
	molecule: str,
	electric_field_list: list,
	jmax_list: list,
	temperature_list: list,
	spin_type: str,
	unit_want: str,
	export_csv: bool = True,
	export_plot: bool = True,
	output_summary_dir: str = "thermo_summary"
):
	"""
	Reads quantum data files and computes thermodynamic properties
	for a specified linear rigid rotor molecule under external fields.

	Parameters:
		base_output_dir (str): Path to the directory containing computed quantum data.
		molecule (str): Name of the linear rigid rotor (e.g., "HF", "HCl").
		electric_field_list (list): List of electric field strengths (in kV/cm).
		jmax_list (list): List of maximum angular momentum quantum numbers used.
		temperature_list (list): List of temperatures (in Kelvin) for thermodynamic evaluation.
		spin_type (str): Type of spin model, default is "spinless".
		unit_want (str): Output unit for thermodynamic quantities ("cm-1" or "J/mol").
		export_csv (bool): Whether to export results as CSV files.
		export_plot (bool): Whether to generate and save plots.
		output_summary_dir (str): Output directory to store summary results.

	Returns:
		None
	"""
	os.makedirs(output_summary_dir, exist_ok=True)

	thermo_dict_by_field = {}
	for jmax, E in product(jmax_list, electric_field_list):

		subdir_name = f"{spin_type}_{molecule}_jmax_{jmax}_field_{E:.2f}kV_per_cm"
		

		# Define filenames
		nc_file_name = f"quantum_data_{subdir_name}.nc"
		nc_file_path = os.path.join(base_output_dir, subdir_name, "data", nc_file_name)

		population_file_name = f"equilibrium_state_population_data_{subdir_name}"
		equilibrium_population_file_path = os.path.join(output_summary_dir, population_file_name)

		thermo_file_name = f"equilibrium_thermodynamic_properties_{subdir_name}"
		equilibrium_properties_file_path = os.path.join(output_summary_dir, thermo_file_name)

		cv_plot_file_name = f"heat_capacity_vs_temperature_plot_{subdir_name}"
		cv_vs_temp_plot_file_path = os.path.join(output_summary_dir, cv_plot_file_name)

		print(f"\n[OK] Checking file: {nc_file_path}")
		if os.path.exists(nc_file_path):
			try:
				with Dataset(nc_file_path, 'r') as nc:
					if "eigenvalues" not in nc.variables:
						print("[WARNING] No eigenvalues found.")
						continue

					eigenval_var = nc.variables["eigenvalues"]
					eigenvalues = np.array(eigenval_var[:])
					print(f"\n[ ] Found {len(eigenvalues)} eigenvalues.")

					# Extract attributes safely
					unit = getattr(eigenval_var, "units", "N/A")
					long_name = getattr(eigenval_var, "long_name", "N/A")

					print(f"[ ] Unit	   : {unit}")
					print(f"[ ] Long name  : {long_name}")


					# Compute thermo data
					thermo_data = compute_thermo_from_eigenvalues(eigenvalues, temperature_list, unit=unit_want)
					
					print("\n[INFO] Summary of important thermodynamic properties:\n")

					for T, entry in thermo_data.items():
						print(f"\n[ ] T = {T} K")
						print(f"[ ] Partition Function = {entry['partition_function']:.4f}")
						print(f"[ ] U = {entry['internal_energy']:.4f} {entry['display_unit']}")
						print(f"[ ] Cv = {entry['heat_capacity']:.4f} {entry['display_cv_unit']}")


					save_thermo_with_Z_and_populations(
						thermo_data=thermo_data,
						temperatures=temperature_list,
						eigenvalues=eigenvalues,
						unit=unit_want,
						txt_path=f"{equilibrium_properties_file_path}.txt",
						csv_path=f"{equilibrium_properties_file_path}.csv",
						save_populations=True,  # Save individual files
						population_dir=f"{equilibrium_population_file_path}"
					)
					"""

					base_name_heat_capacity_vs_temperature_plot = os.path.join(output_summary_dir, filename_heat_capacity_vs_temperature_plot)
					plot_cv_vs_temperature(
						thermo_data=thermo_data,
						unit=unit_want,
						context="Rotational",
						out_path=f"{base_name_heat_capacity_vs_temperature_plot}.png",
					)

					thermo_dict_by_field[(lmax, E)] = thermo_data
					"""

			except Exception as e:
				print(f"[X] Error reading file: {e}")
		else:
			print("[!] File does not exist.\n")

	#plot_cv_surface(
	#	thermo_dict_by_field=thermo_dict_by_field,
	#	temperature_list=temperature_list,
	#	unit=unit_want,
	#	mode="surface",  # or "wireframe"
	#	out_path="cv_surface.png"
	#)

	"""
	plot_cv_surface(
		thermo_dict_by_field=thermo_dict_by_field,
		temperature_list=temperature_list,
		unit="J/mol",
		mode="wireframe",
		out_path="cv_surface_lmax10.png",
		fixed_lmax=10,
		title=r"Rotational Heat Capacity $C_v(T, E)$ for $l_{\max} = 10$"
	)

	plot_cv_heatmap(
		thermo_dict_by_field=thermo_dict_by_field,
		temperature_list=temperature_list,
		unit="J/mol",
		fixed_lmax=10,
		out_path="cv_heatmap_lmax10_annotated.png",
		export_txt=True,
		txt_path="cv_matrix_lmax10.txt"
	)
	"""


	#plot_cv_overlay(
	#	thermo_dict_by_field=thermo_dict_by_field,
	#	unit=unit,
	#	out_path="cv_overlay.png"
	#)




# === Usage ===
read_all_quantum_data_files_with_thermo(
	base_output_dir="/Users/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor/output/",
	molecule="HF",
	electric_field_list=[0.1] + list(range(20, 201, 20)),
	jmax_list=list(range(10, 21, 2)),
	temperature_list=list(range(10, 51, 5)),
	spin_type="spinless",
	unit_want="wavenumber",
	#unit_want="SI",
	export_csv=True,
	export_plot=True,
	output_summary_dir="/Users/tapas/academic-project/results/result_spinless_HF_monomer_in_field"
)

whom()
whoami()
