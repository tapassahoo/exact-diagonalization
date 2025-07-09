import os
from itertools import product
import numpy as np
import pandas as pd
import matplotlib as mpl
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
from scipy.constants import R, k as k_B, N_A, h, c, e as e_charge
import thermodynamics_kelvin as tk
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


k_B_cm = 0.69503476  # Boltzmann constant in cm⁻¹/K
cm_to_J = h * c * 100
cm_to_eV = cm_to_J / e_charge
k_B_cm = k_B / cm_to_J


def check_variable_attribute(nc_path, variable_name=None, attribute_name="units"):
	"""
	General-purpose function to check a specific attribute (default: 'units') 
	of any variable in a NetCDF file.

	Parameters:
		nc_path (str): Path to the NetCDF file.
		variable_name (str or None): Name of the variable to inspect. 
									 If None, lists all variables.
		attribute_name (str): The attribute to retrieve (e.g., 'units', 'long_name').

	Returns:
		str or None: Attribute value if found, else informative message.
	"""
	with Dataset(nc_path, 'r') as nc:
		if variable_name is None:
			print("[i] Variable not specified. Available variables:")
			for var in nc.variables:
				print(f"  - {var}")
			return None

		if variable_name in nc.variables:
			var = nc.variables[variable_name]
			attr_value = getattr(var, attribute_name, None)
			if attr_value is not None:
				print(f"[✓] '{attribute_name}' of variable '{variable_name}': {attr_value}")
				return attr_value
			else:
				print(f"[!] Attribute '{attribute_name}' not found for variable '{variable_name}'.")
				return None
		else:
			print(f"[X] Variable '{variable_name}' not found in file.")
			return None


def read_all_attributes(filename):
	"""Helper function to read and print global and variable-level attributes from a NetCDF file."""
	with Dataset(filename, 'r') as ncfile:
		print("\nGlobal Attributes")
		print("-" * 60)
		for attr in ncfile.ncattrs():
			print(f"{attr:30}: {ncfile.getncattr(attr)}")

		print("\nVariable-wise Attributes")
		print("-" * 60)
		for var_name, var in ncfile.variables.items():
			print(f"\nVariable: {var_name}")
			for attr in var.ncattrs():
				print(f"  {attr:28}: {var.getncattr(attr)}")
	print("\nAttribute inspection complete.\n")


def inspect_netcdf_file(filename):
	"""Inspect variables and their shapes in a NetCDF file."""
	with Dataset(filename, 'r') as ncfile:
		print("Variables in the NetCDF file:")
		variable_names = []
		for var_name, var in ncfile.variables.items():
			shape_str = ", ".join(str(s) for s in var.shape)
			print(f"- {var_name} (dimensions: {shape_str})")
			variable_names.append(var_name)

		print("\nGlobal Attributes in the NetCDF file:")
		for attr in ncfile.ncattrs():
			print(f"- {attr}")

		return variable_names

def read_scalar_like_parameters_with_units(filename, params):
	"""
	Read scalar or scalar-like (single-value) parameters and their units.
	"""
	scalar_params = {}
	with Dataset(filename, "r") as ncfile:
		for param in params:
			if param in ncfile.variables:
				var = ncfile.variables[param]
				try:
					value = var[()] if var.ndim == 0 else var[:].item() if var.size == 1 else None
				except Exception as e:
					print(f"Error reading '{param}': {e}")
					continue

				if value is not None:
					unit = var.getncattr("units") if "units" in var.ncattrs() else "unknown"
					scalar_params[param] = {"value": value, "unit": unit}
			else:
				print(f"Warning: '{param}' not found in the file.")
	return scalar_params

def display_scalar_parameters(params):
	"""Neatly display scalar parameters."""
	if not params:
		print("\nNo scalar-like parameters found.")
		return

	print("\nScalar-like Parameters with Units:")
	print("-" * 55)
	print(f"{'Parameter':<40} {'Value':>8}   {'Unit'}")
	print("-" * 55)
	for name, data in params.items():
		print(f"{name:<40} {data['value']:>8}   {data['unit']}")
	print("-" * 55)

def read_quantum_data(filename, spin_state_name):
	"""
	Read quantum numbers, eigenvalues, and eigenvectors from a NetCDF file.
	
	Parameters:
	- filename: str, path to the NetCDF file
	
	Returns:
	- Tuple of numpy arrays: 
		all_quantum_numbers, spin_state_qn_array, 
		eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors (complex)
	"""
	with Dataset(filename, "r") as ncfile:
		# Read all quantum numbers
		all_quantum_numbers = ncfile.variables["all_quantum_numbers"][:]
		
		# Automatically determine spin state from available variables
		spin_state_qn_array = ncfile.variables[f"{spin_state_name}_quantum_numbers"][:]
		
		# Read eigenvalues and eigenvectors
		eigenvalues = ncfile.variables["eigenvalues"][:]
		eigenvectors_real = ncfile.variables["eigenvectors_real"][:]
		eigenvectors_imag = ncfile.variables["eigenvectors_imag"][:]

		# Reconstruct complex eigenvectors
		eigenvectors = eigenvectors_real + 1j * eigenvectors_imag

	return all_quantum_numbers, spin_state_qn_array, eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors



def read_all_quantum_data_files(
	base_output_dir,
	dipole_moment_D,
	electric_field_kVcm_list,
	max_angular_momentum_list,
	spin_type="spinless"
):
	"""
	Read NetCDF data for various configurations of an HF monomer in an electric field.

	Parameters:
		base_output_dir (str): Base path where output directories are stored.
		dipole_moment_D (float): Dipole moment in Debye.
		electric_field_kVcm_list (list): Electric field strengths in kV/cm.
		max_angular_momentum_list (list): Values of maximum angular momentum quantum number (lmax).
		spin_type (str): Rotor spin type: 'spinless', 'ortho', or 'para'.
	"""
	for lmax, E in product(max_angular_momentum_list, electric_field_kVcm_list):
		# Define grid size based on lmax
		theta_grid_count = 2 * lmax + 5
		phi_grid_count = 2 * theta_grid_count + 5

		# Clean subdirectory and filename strings (avoid dots)
		subdir = (
			f"{spin_type}_HF_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm"
		).replace(".", "_")

		filename = (
			f"quantum_data_HF_{spin_type}_isomer_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm_"
			f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}.nc"
		)

		file_path = os.path.join(base_output_dir, subdir, filename)
		check_variable_attribute(file_path, variable_name="eigenvalues", attribute_name="units")
		check_variable_attribute(file_path, variable_name="eigenvalues", attribute_name="long_name")



		"""
		print(f"\nChecking file: {file_path}")
		if os.path.exists(file_path):
			try:
				read_all_attributes(file_path)
				all_variables = inspect_netcdf_file(file_path)
				scalar_data = read_scalar_like_parameters_with_units(file_path, all_variables)
				display_scalar_parameters(scalar_data)
				with Dataset(file_path, 'r') as nc:
					print("Variable Summary")
					print("-" * 60)
					for var in nc.variables:
						shape = nc.variables[var].shape
						dtype = nc.variables[var].dtype
						print(f"  {var:30}: shape = {shape}, dtype = {dtype}")
				print("File read successfully.\n")
			except Exception as e:
				print(f"Error reading file: {e}")
		else:
			print("File does not exist.\n")
		"""

def compute_thermo_from_eigenvalues(eigenvalues_kelvin, temperature_list, unit="Kelvin"):
	"""
	Compute thermodynamic quantities (Boltzmann populations, partition function, internal energy 
	and heat capacity) from energy eigenvalues given in Kelvin.

	Parameters:
	-----------
	eigenvalues_kelvin : array_like
		Array of energy eigenvalues in Kelvin.
	
	temperature_list : array_like
		List or array of temperatures in Kelvin at which thermodynamic properties are evaluated.

	unit : str, optional
		Output unit for energy and heat capacity. Choose from:
		- "Kelvin" (default)
		- "J/mol"

	Returns:
	--------
	dict
		Dictionary with temperatures as keys and a sub-dictionary as values containing:
		- "Populations" : Normalized Boltzmann populations
		- "Z" : Partition function
		- "U (unit)" : Internal energy
		- "Cv (unit/K)" : Heat capacity
	"""
	# Convert input to numpy array
	E = np.array(eigenvalues_kelvin, dtype=np.float64)
	if E.ndim != 1:
		raise ValueError("Energy eigenvalues must be a 1D array.")

	thermo = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"Temperature must be positive. Received: {T}")

		beta = 1.0 / T

		# Apply zero-point shift for numerical stability
		E_shifted = E - np.min(E)
		weights = np.exp(-beta * E_shifted)
		Z = np.sum(weights)
		P = weights / Z  # Normalized Boltzmann populations

		# Use original energies for observables
		E_avg = np.sum(P * E)
		E2_avg = np.sum(P * E**2)
		Cv = beta**2 * (E2_avg - E_avg**2)

		# Unit conversion
		if unit == "Kelvin":
			U_out = E_avg
			Cv_out = Cv
		elif unit == "J/mol":
			U_out = E_avg * R 
			Cv_out = Cv * R
		else:
			raise ValueError("Invalid unit. Choose from 'Kelvin' or 'J/mol'.")

		# Store results
		thermo[T] = {
			"Populations": P,
			"Z": Z,
			f"U ({unit})": U_out,
			f"Cv ({unit}/K)": Cv_out,
		}

	return thermo

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
	unit="Kelvin",
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
			Z = thermo_data[T]["Z"]
			U = thermo_data[T][f"U ({unit})"]
			Cv = thermo_data[T][f"Cv ({unit}/K)"]
			f.write(f"{T:15.1f}  {U:15.6f}  {Cv:15.6f}  {Z:15.6f}\n")
	print(f"[✓] TXT summary saved: {txt_path}")

	# -----------------------------------
	# Save CSV summary
	# -----------------------------------
	df = pd.DataFrame({
		"Temperature (K)": sorted(temperatures),
		f"U ({unit})": [thermo_data[T][f"U ({unit})"] for T in sorted(temperatures)],
		f"Cv ({unit}/K)": [thermo_data[T][f"Cv ({unit}/K)"] for T in sorted(temperatures)],
		"Partition Function Z": [thermo_data[T]["Z"] for T in sorted(temperatures)]
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
	base_output_dir,
	dipole_moment_D,
	electric_field_kVcm_list,
	max_angular_momentum_list,
	temperature_list,
	spin_type="spinless",
	unit_want="Kelvin",
	export_csv=True,
	export_plot=True,
	output_summary_dir="thermo_summary"
):
	os.makedirs(output_summary_dir, exist_ok=True)

	thermo_dict_by_field = {}
	for lmax, E in product(max_angular_momentum_list, electric_field_kVcm_list):
		theta_grid_count = 2 * lmax + 5
		phi_grid_count = 2 * theta_grid_count + 5

		subdir = (
			f"{spin_type}_HF_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm"
		).replace(".", "_")

		filename = (
			f"quantum_data_HF_{spin_type}_isomer_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm_"
			f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}.nc"
		)

		filename_thermo_data = (
			f"equilibrium_properties_data_HF_{spin_type}_isomer_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm_"
			f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}"
		)

		filename_heat_capacity_vs_temperature_plot = (
			f"heat_capacity_vs_temperature_plot_HF_{spin_type}_isomer_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm_"
			f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}"
		)

		filename_population_data = (
			f"equilibrium_population_data_HF_{spin_type}_isomer_lmax_{lmax}_"
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{E:.2f}kVcm_"
			f"theta_grid_{theta_grid_count}_phi_grid_{phi_grid_count}"
		)

		file_path = os.path.join(base_output_dir, subdir, filename)

		print(f"\n[✓] Checking file: {file_path}")
		if os.path.exists(file_path):
			try:
				with Dataset(file_path, 'r') as nc:
					if "eigenvalues" not in nc.variables:
						print("[!] No eigenvalues found.")
						continue

					eigenval_var = nc.variables["eigenvalues"]
					eigenvalues = np.array(eigenval_var[:])
					print(f"  Found {len(eigenvalues)} eigenvalues.")

					# Extract attributes safely
					unit = getattr(eigenval_var, "units", "N/A")
					long_name = getattr(eigenval_var, "long_name", "N/A")

					print(f"  Unit	   : {unit}")
					print(f"  Long name  : {long_name}")


					# Compute thermo data
					thermo_data = compute_thermo_from_eigenvalues(eigenvalues, temperature_list, unit=unit_want)

					"""
					for T in thermo_data:
						print(f"\nT = {T} K")
						print(f"U = {thermo_data[T]['U (J/mol)']:.4f}")
						print(f"Cv = {thermo_data[T]['Cv (J/mol/K)']:.4f}")
						print(f"Populations: {thermo_data[T]['Populations']}")
					"""

					# Output file prefix
					base_name_equilibrium_properties = os.path.join(output_summary_dir, filename_thermo_data)
					base_name_equilibrium_population = os.path.join(output_summary_dir, filename_population_data)

					save_thermo_with_Z_and_populations(
						thermo_data=thermo_data,
						temperatures=temperature_list,
						eigenvalues=eigenvalues,
						unit=unit_want,
						txt_path=f"{base_name_equilibrium_properties}.txt",
						csv_path=f"{base_name_equilibrium_properties}.csv",
						save_populations=True,  # Save individual files
						population_dir=f"{base_name_equilibrium_population}"
					)

					base_name_heat_capacity_vs_temperature_plot = os.path.join(output_summary_dir, filename_heat_capacity_vs_temperature_plot)
					plot_cv_vs_temperature(
						thermo_data=thermo_data,
						unit=unit_want,
						context="Rotational",
						out_path=f"{base_name_heat_capacity_vs_temperature_plot}.png",
					)

					thermo_dict_by_field[(lmax, E)] = thermo_data

			except Exception as e:
				print(f"[X] Error reading file: {e}")
		else:
			print("[!] File does not exist.\n")

	# Assume:
	# thermo_dict_by_field = {
	#	 0.0: thermo_0kVcm,
	#	 0.1: thermo_01kVcm,
	#	 ...
	# }

	#plot_cv_surface(
	#	thermo_dict_by_field=thermo_dict_by_field,
	#	temperature_list=temperature_list,
	#	unit=unit_want,
	#	mode="surface",  # or "wireframe"
	#	out_path="cv_surface.png"
	#)

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


	#plot_cv_overlay(
	#	thermo_dict_by_field=thermo_dict_by_field,
	#	unit=unit,
	#	out_path="cv_overlay.png"
	#)




# === Usage ===
"""
read_all_quantum_data_files(
	base_output_dir="/Users/tapas/academic-project/outputs/output_spinless_HF_monomer_in_field",
	dipole_moment_D=1.83,
	electric_field_kVcm_list=[0.1],# + list(range(10, 201, 10)),
	max_angular_momentum_list=list(range(10, 11, 5)),
	spin_type="spinless"
)

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

filename="output/data/quantum_data_HCl_spinless_isomer_lmax_20_dipole_moment_1.80D_electric_field_200.00kVcm.nc"
read_all_attributes(filename)
inspect_netcdf_file(filename)

whom()
whoami()

#spin_state_name = "spinless"
#all_quantum_numbers, quantum_numbers_for_spin_state, eigenvalues, eigenvectors_real, eigenvectors_imag, eigenvectors = read_quantum_data(filename, spin_state_name)
