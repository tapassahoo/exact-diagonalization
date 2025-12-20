from datetime import datetime
import os
import math
from pathlib import Path
from itertools import product
import numpy as np
from netCDF4 import Dataset
from typing import Optional, Union
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.constants import R as GAS_CONSTANT_J_PER_MOL_K
from scipy.constants import k as BOLTZMANN_J_PER_K
import warnings
import itertools


from monomer_linear_rotor.utils import (
	wavenumber_to_joules_per_mole,
)
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom

def plot_cv_vs_temperature(
	thermo_data: dict,
	unit: str = None,				# <-- Optional manual override
	out_path: str = None,
	title: str = None,
	context: str = "Rotational"
):
	"""
	Plot heat capacity (Cv) vs Temperature using LaTeX-rendered labels.

	Parameters:
		thermo_data (dict): Dictionary with temperature as keys and Cv data as values.
		unit (str): Optional override for Cv unit (e.g., 'cm^{-1}', 'J/mol'). If None, uses value from thermo_data.
		out_path (str): If provided, saves the plot to this path; otherwise displays the plot.
		title (str): Custom plot title. If None, a default one is generated.
		context (str): Physical context such as "Rotational", "Vibrational", etc.
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

	if not thermo_data:
		print("[X] No thermodynamic data to plot.")
		return

	# Extract units
	first_T = next(iter(thermo_data))
	final_unit = thermo_data[first_T]["display_cv_unit"]

	# Prepare data
	T_vals = sorted(thermo_data.keys())
	Cv_vals = [thermo_data[T]["heat_capacity"] for T in T_vals]

	# Default title
	if title is None:
		title = rf"{context} Heat Capacity $C_V$ vs Temperature $T$"

	# Plotting
	plt.figure(figsize=(6, 4))
	plt.plot(T_vals, Cv_vals, 'o-', label=rf"$C_V$ ($\mathrm{{{final_unit}}}$/K)")
	plt.xlabel(r"Temperature $T$ (K)")
	plt.ylabel(rf"$C_V$ ($\mathrm{{{final_unit}}}$/K)")
	plt.title(title)
	plt.grid(True)
	plt.legend()
	plt.tight_layout()

	if out_path:
		plt.savefig(out_path, dpi=300)
		print(f"[✓] Plot saved to: {out_path}")
	else:
		plt.show()

	plt.close()

def plot_cv_comparison(thermo_dict_by_molecule, get_temperature_list, unit_want, out_path):
	"""
	Plots heat capacity vs temperature for multiple molecules together.

	Parameters:
		thermo_dict_by_molecule (dict): { molecule: {(jmax, E): thermo_data} }
		get_temperature_list (function): Function to fetch temperature list for a molecule.
		unit_want (str): Unit for Cv display.
		out_path (str or Path): Path to save combined plot.
	"""

	plt.figure(figsize=(12, 6))

	color_cycle = plt.cm.tab10.colors
	line_styles = ["-", "--", "-.", ":"]
	markers = ["*", "p", "s", "o", "v"]

	for mol_idx, (molecule, thermo_dict) in enumerate(thermo_dict_by_molecule.items()):
		temperature_list = get_temperature_list(molecule)

		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		color = color_cycle[mol_idx % len(color_cycle)]
		mk = markers[mol_idx % len(markers)]

		for curve_idx, ((jmax, E), thermo_data) in enumerate(thermo_dict.items()):
			cv_values = [thermo_data[T]["heat_capacity"] for T in temperature_list]
			unit_cv = thermo_data[temperature_list[0]]["display_cv_unit"]

			ls = line_styles[curve_idx % len(line_styles)]

			plt.plot(
				temperature_list,
				cv_values,
				linestyle=ls,
				marker=mk,
				color=color,
				label=rf"{molecule} ($J_{{\max}}={jmax}$, $E={E:.2f}\,\mathrm{{kV/cm}}$)"
			)

	# Assign consistent colors per molecule
	"""
	color_cycle = plt.cm.tab10.colors  
	line_styles = ["-", "--", "-.", ":"]
	markers = ["o", "s", "D", "^", "v"]

	for mol_idx, (molecule, thermo_dict) in enumerate(thermo_dict_by_molecule.items()):
		temperature_list = get_temperature_list(molecule)

		# Flatten if nested
		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		color = color_cycle[mol_idx % len(color_cycle)]
		style_iter = itertools.product(line_styles, markers)

		for (jmax, E), thermo_data in thermo_dict.items():
			cv_values = [thermo_data[T]["heat_capacity"] for T in temperature_list]
			unit_cv = thermo_data[temperature_list[0]]["display_cv_unit"]

			ls, mk = next(style_iter)

			plt.plot(
				temperature_list,
				cv_values,
				linestyle=ls,
				marker=mk,
				color=color,
				label=rf"{molecule} (Jmax={jmax}, E={E:.2f} kV/cm)"
			)
	"""

	# X-axis ticks
	xticks = np.arange(0, 101, 10)
	plt.xticks(xticks, fontsize=18)
	plt.xlabel("Temperature (K)", fontsize=18)
	safe_unit = unit_cv.replace("^-1", "$^{-1}$")  
	plt.ylabel(rf"Heat Capacity ($C_V$) [{safe_unit}]", fontsize=18)
	#plt.tick_params(axis='x', labelsize=18, colors='red')
	# Tick parameters
	plt.tick_params(axis='both', labelsize=18)
	plt.xlim(-0.5, 100.5)

	#plt.title("Heat Capacity vs Temperature for Linear Rotors", fontsize=14)
	plt.legend(fontsize=18, loc="best")
	plt.grid(True, ls="--", alpha=0.6)
	plt.tight_layout()

	# Save first, then show
	plt.savefig(out_path, dpi=300)
	print("")
	print(f"[INFO] Combined Cv plot saved: {out_path}")

if False:
	def plot_cv_comparison(thermo_dict_by_molecule, get_temperature_list, unit_want, out_path):
		"""
		Plots heat capacity vs temperature for multiple molecules together.
		
		Parameters:
			thermo_dict_by_molecule (dict): { molecule: {(jmax, E): thermo_data} }
			temperature_list (list): List of temperatures (K).
			unit_want (str): Unit for Cv display.
			out_path (str or Path): Path to save combined plot.
		"""

		plt.figure(figsize=(8, 6))
		for molecule, thermo_dict in thermo_dict_by_molecule.items():
			temperature_list = get_temperature_list(molecule)

			# Flatten if nested
			if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
				temperature_list = temperature_list[0]

			#print(f"[DEBUG] {molecule} temperature list: {temperature_list}")

			first_key = next(iter(thermo_dict))
			thermo_data = thermo_dict[first_key]

			cv_values = [thermo_data[T]["heat_capacity"] for T in temperature_list]
			#print(f"[DEBUG] {molecule} Cv values: {cv_values}")
			unit_cv = thermo_data[temperature_list[0]]["display_cv_unit"]

			plt.plot(
				temperature_list,
				cv_values,
				marker="o",
				label=f"{molecule}"
			)

		plt.xlabel("Temperature (K)")

		# Fix LaTeX formatting for unit safely
		safe_unit = unit_cv.replace("^-1", "$^{-1}$")
		plt.ylabel(f"Heat Capacity (Cv) [{safe_unit}]")

		plt.title("Heat Capacity vs Temperature for Linear Rotors")
		plt.legend()
		#plt.grid(True, ls="--", alpha=0.6)
		plt.tight_layout()

		# Save first, then show
		plt.savefig(out_path, dpi=300)
		print(f"[INFO] Combined Cv plot saved: {out_path}")

		plt.show()
		plt.close()

def compute_thermo_from_eigenvalues(eigenvalues, temperature_list, unit):
	"""
	Compute thermodynamic properties (Z, populations, U, Cv) from energy eigenvalues,
	and report the index and energy at which Boltzmann convergence is reached.

	Parameters
	----------
	eigenvalues : array_like
		1D array of energy eigenvalues in wavenumber units (cm⁻¹).

	temperature_list : array_like
		List or array of temperatures (in Kelvin) for which properties are computed.

	unit : {'wavenumber', 'SI'}
		Desired output unit system:
			- 'wavenumber' : Energy in cm⁻¹ and heat capacity in cm⁻¹/K
			- 'SI'		 : Energy in J/mol and heat capacity in J/mol·K

	Returns
	-------
	dict
		Dictionary keyed by temperature (in K), each entry containing:
			- temperature_K		: Temperature in Kelvin
			- unit				: 'wavenumber' or 'SI'
			- display_unit		: Unit for U
			- display_cv_unit	: Unit for Cv
			- beta				: 1 / (kB·T) in cm⁻¹⁻¹
			- partition_function: Canonical partition function Z
			- populations		: Normalized Boltzmann populations
			- internal_energy	: Mean energy (U)
			- heat_capacity		: Heat capacity (Cv)
			- levels_used		: Number of energy levels included
			- convergence_energy: Energy at which convergence was met (in cm⁻¹)
			- convergence_index : Index where threshold was first met
	"""

	energies = np.atleast_1d(np.asarray(eigenvalues, dtype=np.float64)).flatten()
	if energies.ndim != 1:
		raise ValueError("Eigenvalues must be a one-dimensional array.")
	if unit not in {"wavenumber", "SI"}:
		raise ValueError("Unit must be either 'wavenumber' or 'SI'.")

	BOLTZMANN_CM_INV_PER_K = 0.69503476		   # k_B = 0.69503476 cm^-1/K
	threshold = 1e-100

	results = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"Temperature must be > 0 K. Got: {T}")

		beta = 1.0 / (BOLTZMANN_CM_INV_PER_K * T) # cm
		E_shifted = energies - np.min(energies)   # cm^-1

		weights = []
		E_used = []
		convergence_index = None
		converged = False

		for i, Ei in enumerate(E_shifted):
			wi = np.exp(-beta * Ei)			   # unitless
			if wi <= threshold:
				convergence_index = i
				converged = True
				break
			weights.append(wi)
			E_used.append(Ei)

		if not converged:
			convergence_index = len(energies)
			warnings.warn(
				f"Convergence not reached at T = {T:.2f} K. Boltzmann factor did not fall below {threshold}.",
				RuntimeWarning
			)

		weights = np.array(weights)
		E_used = np.array(E_used)
		Z = np.sum(weights)
		populations = weights / Z if Z > 0 else np.zeros_like(weights)

		energies_used_orig = energies[:convergence_index]
		E_avg = np.dot(populations, energies_used_orig)
		E2_avg = np.dot(populations, energies_used_orig**2)
		Cv_cm1 = BOLTZMANN_CM_INV_PER_K * beta**2 * (E2_avg - E_avg**2) # cm^-1/K

		if unit == "wavenumber":
			U_out = E_avg
			Cv_out = Cv_cm1
			display_unit = "cm^-1"
			display_cv_unit = "cm^-1/K"
		else:
			U_out = wavenumber_to_joules_per_mole(E_avg)
			Cv_out = wavenumber_to_joules_per_mole(Cv_cm1)
			display_unit = "J/mol"
			display_cv_unit = "J/mol·K"

		results[T] = {
			"temperature_K": T,
			"unit": unit,
			"display_unit": display_unit,
			"display_cv_unit": display_cv_unit,
			"beta": beta,
			"partition_function": Z,
			"populations": populations,
			"internal_energy": U_out,
			"heat_capacity": Cv_out,
			"levels_used": convergence_index,
			"convergence_energy": energies[convergence_index] if convergence_index < len(energies) else None,
			"convergence_index": convergence_index
		}

	return results

def save_thermo_with_Z_and_populations(
	thermo_data: dict,
	temperatures: list,
	eigenvalues: np.ndarray,
	unit: str = "wavenumber",
	txt_path: str = "thermo_summary.txt",
	csv_path: str = "thermo_summary.csv",
	save_populations: bool = False,
	population_dir: str = "populations_txt"
):
	"""
	Save thermodynamic data (U, Cv, Z) to TXT and CSV files, and optionally
	save population distributions for each temperature.

	Parameters:
		thermo_data (dict): Thermodynamic data keyed by temperature.
		temperatures (list): List of temperatures (K).
		eigenvalues (np.ndarray): Array of eigenvalues.
		unit (str): Unit of energy (e.g., 'cm-1', 'J/mol').
		txt_path (str): Output path for TXT summary.
		csv_path (str): Output path for CSV summary.
		save_populations (bool): If True, saves per-temperature population files.
		population_dir (str): Directory to store population text files.
	"""

	# -------------------------
	# Extract display units safely
	# -------------------------
	try:
		first_T = next(iter(thermo_data))
		display_unit = thermo_data[first_T].get("display_unit", "unit")
		display_cv_unit = thermo_data[first_T].get("display_cv_unit", "unit")
	except Exception as e:
		print(f"[X] Error accessing display units: {e}")
		display_unit = "unit"
		display_cv_unit = "unit"

	# -------------------------
	# Write TXT summary
	# -------------------------
	header_line = (
		f"{'Temperature (K)':>15}  "
		f"{f'U ({display_unit})':>20}  "
		f"{f'Cv ({display_cv_unit})':>20}  "
		f"{'Z':>15}\n"
	)
	divider = "-" * len(header_line.strip())

	try:
		with open(txt_path, "w") as f:
			f.write(f"# Thermodynamic summary generated on {datetime.now():%Y-%m-%d %H:%M:%S}\n")
			f.write("# " + header_line)
			f.write("#" + divider + "\n")

			for T in sorted(temperatures):
				entry = thermo_data.get(T, {})
				Z = entry.get("partition_function")
				U = entry.get("internal_energy")
				Cv = entry.get("heat_capacity")

				Z_str = f"{Z:15.6f}" if isinstance(Z, (int, float)) else f"{'N/A':>15}"
				U_str = f"{U:20.6f}" if isinstance(U, (int, float)) else f"{'N/A':>20}"
				Cv_str = f"{Cv:20.6f}" if isinstance(Cv, (int, float)) else f"{'N/A':>20}"

				f.write(f"{T:15.1f}  {U_str}  {Cv_str}  {Z_str}\n")

		print(f"[INFO] TXT summary saved: {txt_path}")
	except Exception as e:
		print(f"[X] Failed to write TXT summary: {e}")

	# -------------------------
	# Write CSV summary
	# -------------------------
	try:
		data_dict = {
			"Temperature (K)": [],
			f"U ({display_unit})": [],
			f"Cv ({display_cv_unit})": [],
			"Partition Function Z": []
		}

		for T in sorted(temperatures):
			entry = thermo_data.get(T, {})
			data_dict["Temperature (K)"].append(T)
			data_dict[f"U ({display_unit})"].append(entry.get("internal_energy"))
			data_dict[f"Cv ({display_cv_unit})"].append(entry.get("heat_capacity"))
			data_dict["Partition Function Z"].append(entry.get("partition_function"))

		df = pd.DataFrame(data_dict)
		df.to_csv(csv_path, index=False)
		print(f"[INFO] CSV summary saved: {csv_path}")
	except Exception as e:
		print(f"[X] Failed to write CSV summary: {e}")

	# -------------------------
	# Save population distributions (optional)
	# -------------------------
	if save_populations:
		try:
			pop_dir = Path(population_dir)
			pop_dir.mkdir(parents=True, exist_ok=True)
			eigenvalues = np.asarray(eigenvalues)

			for T in sorted(temperatures):
				entry = thermo_data.get(T, {})
				populations = entry.get("populations")
				if populations is None:
					continue

				pop_file_path = pop_dir / f"populations_T_{T:.1f}K.txt"
				with open(pop_file_path, "w") as pf:
					pf.write(f"# Population distribution at T = {T:.1f} K\n")
					pf.write(f"# {'Index':>6}  {'Energy (' + display_unit + ')':>20}  {'P_i':>15}\n")
					pf.write("#" + "-" * 55 + "\n")
					for i, (E_i, P_i) in enumerate(zip(eigenvalues, populations)):
						pf.write(f"{i:6d}  {E_i:20.6f}  {P_i:20.6e}\n")
				print(f"[INFO] Populations saved: {pop_file_path}")

		except Exception as e:
			print(f"[X] Failed to save population files: {e}")

def read_all_quantum_data_files_with_thermo(
	quantum_data_root_dir: str,
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
		quantum_data_root_dir (str): Root directory containing computed quantum data subdirectories for different field and Jmax values.
		molecule (str): Name of the linear rotor (e.g., 'HF', 'HCl').
		electric_field_list (list): List of electric field strengths (in kV/cm).
		jmax_list (list): List of maximum J values used in the calculations.
		temperature_list (list): List of temperatures (in K) for thermodynamic analysis.
		spin_type (str): 'spinless', 'ortho', 'para', etc.
		unit_want (str): Output unit for thermodynamic quantities ("cm-1" or "J/mol").
		export_csv (bool): Export thermodynamic data to CSV (default: True).
		export_plot (bool): Generate heat capacity plots (default: True).
		output_summary_dir (str): Directory to store summary outputs (default: "thermo_summary").

	Returns:
		dict: A dictionary mapping (jmax, E_field) → thermo_data
	"""

	# Ensure output base directory is a Path object
	output_base_dir = Path(output_summary_dir)

	# Construct a descriptive subdirectory name for summaries
	summary_subdir = f"{spin_type}_{molecule}_monomer_in_electric_field"
	summary_output_dir = output_base_dir / summary_subdir

	# Check if directory exists
	if summary_output_dir.exists():
		print(f"[INFO] Output directory already exists: {summary_output_dir.resolve()}\n")
	else:
		# Create the output directory if it does not exist
		summary_output_dir.mkdir(parents=True, exist_ok=True)
		print(f"[INFO] Output directory created: {summary_output_dir.resolve()}\n")

	# Initialize dictionary to hold thermo data by (jmax, E)
	thermo_dict_by_field = {}

	# Iterate over combinations of jmax and electric field strengths
	for jmax, E in product(jmax_list, electric_field_list):
		# Construct data subdirectory name
		data_subdir = f"{spin_type}_{molecule}_jmax_{jmax}_field_{E:.2f}kV_per_cm"

		# Full path to the .nc file in quantum data root directory
		nc_file_path = Path(quantum_data_root_dir) / data_subdir / "data" / f"quantum_data_{data_subdir}.nc"

		# Example placeholder for handling (to be continued...)
		print(f"\n[INFO] Looking for file: {nc_file_path}\n")

		if not nc_file_path.exists():
			print("[!] File does not exist.")
			continue

		try:
			with Dataset(nc_file_path, 'r') as nc:
				if "eigenvalues" not in nc.variables:
					print("[WARNING] 'eigenvalues' variable not found in the file.")
					continue

				eigenval_var = nc.variables["eigenvalues"]
				eigenvalues = np.array(eigenval_var[:])
				unit_from_file = getattr(eigenval_var, "units", "unknown")
				label_from_file = getattr(eigenval_var, "long_name", "eigenvalues")

				print(f"[ ] {'Found':<12}: {len(eigenvalues)} eigenvalues")
				print(f"[ ] {'Unit':<12}: {unit_from_file}")
				print(f"[ ] {'Description':<12}: {label_from_file}")

				# Compute thermodynamic properties
				thermo_data = compute_thermo_from_eigenvalues(
					eigenvalues=eigenvalues,
					temperature_list=temperature_list,
					unit=unit_want
				)

				# Print summary
				print("\n[INFO] Thermodynamic Summary:")
				for T in temperature_list:
					entry = thermo_data[T]
					print(f"\n[ ] {'T':<22}= {T} K")
					print(f"[ ] {'levels_used':<22}= {entry['levels_used']}")
					#print(f"[ ] {'convergence_energy':<22}= {entry['convergence_energy']} {entry['display_unit']}")
					 # Convergence energy with conditional unit display
					convergence_energy = entry.get("convergence_energy")
					display_unit = entry.get("display_unit", "")
					if isinstance(convergence_energy, (int, float)) and not math.isnan(convergence_energy):
						print(f"[ ] {'convergence_energy':<22}= {convergence_energy:.6f} {display_unit}")
					else:
						print(f"[ ] {'convergence_energy':<22}= N/A")
					print(f"[ ] {'convergence_index':<22}= {entry['convergence_index']}")
					print(f"[ ] {'Z':<22}= {entry['partition_function']:.4f}")
					print(f"[ ] {'U':<22}= {entry['internal_energy']:.4f} {entry['display_unit']}")
					print(f"[ ] {'Cv':<22}= {entry['heat_capacity']:.4f} {entry['display_cv_unit']}\n")


				file_prefix = summary_output_dir / f"equilibrium_thermodynamic_properties_{data_subdir}"
				pop_dir = summary_output_dir / f"equilibrium_state_population_data_{data_subdir}"
				plot_path = summary_output_dir / f"heat_capacity_vs_temperature_plot_{data_subdir}.png"

				# Export to file
				if export_csv:
					save_thermo_with_Z_and_populations(
						thermo_data=thermo_data,
						temperatures=temperature_list,
						eigenvalues=eigenvalues,
						unit=unit_want,
						txt_path=str(file_prefix) + ".txt",
						csv_path=str(file_prefix) + ".csv",
						save_populations=True,
						population_dir=str(pop_dir)
					)

				if export_plot:
					plot_cv_vs_temperature(
						thermo_data=thermo_data,
						unit=unit_want,
						context="Rotational",
						out_path=plot_path
					)

				# Store in output dictionary
				thermo_dict_by_field[(jmax, E)] = thermo_data

		except Exception as e:
			print(f"[X] Error reading or processing file: {e}")

	return thermo_dict_by_field
