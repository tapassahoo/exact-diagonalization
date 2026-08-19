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
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import R as GAS_CONSTANT_J_PER_MOL_K
from scipy.constants import k as BOLTZMANN_J_PER_K
import warnings
import itertools
from termcolor import colored
from scipy.special import lpmv, gammaln
from numpy.polynomial.legendre import leggauss

from monomer_linear_rotor.molecule_data import MOLECULE_DATA

from monomer_linear_rotor.utils import convert_dipole_field_energy_to_cm_inv


from monomer_linear_rotor.utils import (
	wavenumber_to_joules_per_mole,
	set_plot_style,
)

from monomer_linear_rotor.hamiltonian import (
	rotational_energy_levels,
)

from pkg_utils.utils import whoami
from pkg_utils.config import *
from pkg_utils.env_report import whom


# ============================================================
# CONSTANT
# ============================================================

# Boltzmann constant in cm^{-1} K^{-1}
KB_CM_INV_PER_K = 0.69503476

styles = {
	"HF":  {"linestyle": "-",  "marker": "o"},
	"HCl": {"linestyle": "--", "marker": "s"},
	"HBr": {"linestyle": "-.", "marker": "D"},
	"HI":  {"linestyle": ":",  "marker": "^"},
}


def compute_rotational_levels_cum(
	B,
	T=None,
	J_max=2000,
	tol=1e-100,
	return_dict=False,
	display=False
):
	"""
	Compute rotational energy levels with cumulative-population truncation.

	Parameters
	----------
	B : float
		Rotational constant (cm⁻¹)
	T : float, optional
		Temperature (K). If None → no Boltzmann statistics
	J_max : int
		Maximum J (used if T is None)
	tol : float
		Missing population tolerance (1 - cumulative cutoff)
		e.g., tol=1e-6 → retain 99.9999% population
	return_dict : bool
		If True, also return dictionaries
	display : bool
		If True, print formatted table

	Returns
	-------
	J : ndarray
	E : ndarray
	p : ndarray or None
	cum : ndarray or None
	(optional dicts)
	"""

	k_B_cm = 0.69503476  # cm⁻¹/K

	# ---- Case 1: No temperature ----
	if T is None:
		J = np.arange(J_max + 1)
		E = B * J * (J + 1)
		p = None
		cum = None

	# ---- Case 2: With temperature ----
	else:
		# Estimate upper bound
		J_peak = int(max(0, np.sqrt(k_B_cm * T / (2 * B)) - 0.5))
		J_max_eff = int(J_peak * 6 + 10)

		J_full = np.arange(J_max_eff + 1)
		E_full = B * J_full * (J_full + 1)

		# Boltzmann weights
		w = (2 * J_full + 1) * np.exp(-E_full / (k_B_cm * T))

		# Normalize → probabilities
		Z = np.sum(w)
		p_full = w / Z

		# Cumulative population
		cum_full = np.cumsum(p_full)

		# Find truncation index
		cutoff_idx = np.searchsorted(cum_full, 1 - tol)

		# Slice
		J = J_full[:cutoff_idx + 1]
		E = E_full[:cutoff_idx + 1]
		p = p_full[:cutoff_idx + 1]
		cum = cum_full[:cutoff_idx + 1]

	# ---- Optional dictionaries ----
	if return_dict:
		energies = dict(zip(J, E))
		pop_dict = dict(zip(J, p)) if p is not None else None
		cum_dict = dict(zip(J, cum)) if cum is not None else None

	# ---- Display (optional, separated for HPC cleanliness) ----
	if display:
		print(colored("\nRotational energy levels of a rigid rotor",
					  HEADER_COLOR, attrs=['bold', 'underline']))

		if T is not None:
			print(f"Temperature = {T} K")
			print(f"Retained population = {cum[-1]:.8f}")

			print(f"\n{'J':<5}{'Energy':>12}{'Pop':>15}{'Cumulative':>15}")
			print("=" * 50)

			for j, e, pj, cj in zip(J, E, p, cum):
				print(f"{j:<5}{e:>12.6f}{pj:>15.6e}{cj:>15.6f}")
		else:
			print(f"\n{'J':<5}{'Energy (cm^-1)':>15}")
			print("=" * 22)
			for j, e in zip(J, E):
				print(f"{j:<5}{e:>15.6f}")

	# ---- Return ----
	if return_dict:
		return J, E, p, cum, energies, pop_dict, cum_dict
	else:
		return J, E, p, cum

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
	set_plot_style()

	num_molecules = len(thermo_dict_by_molecule)	
	# -----------------------------
	# Figure setup
	# -----------------------------
	if num_molecules != 1:
		fig, ax = plt.subplots(figsize=(8, 6))

	else:
		# Create figure with 2 row, 1 columns
		fig, axs = plt.subplots(
			2, 1,
			figsize=(8, 12),
		)
	# Colorblind-friendly palette (Okabe–Ito)
	color_cycle = [
		"#0072B2",  # Blue
		"#D55E00",  # Vermillion
		"#009E73",  # Bluish green
		"#CC79A7",  # Reddish purple
	]

	line_styles = [   
		(0, ()),		   # solid
		(0, (5, 1)),	   # densely dashed
		(0, (3, 1, 1, 1)), # densely dashdotted
		(0, (1, 1)),	   # densely dotted
	]

	# Open markers
	markers = ["o", "s", "^", "D"]

	# Rotational heat capacity (constant)
	cv_rot_equipartition_theorem = 0.695  # cm^-1 K^-1

	# Create an array of same size
	temperature_list = get_temperature_list("heat_capacity")
	cv_array = np.full_like(temperature_list, cv_rot_equipartition_theorem)

	for mol_idx, (molecule, thermo_dict) in enumerate(thermo_dict_by_molecule.items()):

		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		color = color_cycle[mol_idx % len(color_cycle)]
		mk = markers[mol_idx % len(markers)]

		if num_molecules == 1:

			# Get J and level energies
			J_num, energies_levels = rotational_energy_levels(MOLECULE_DATA[molecule]["B_const"], 60, display=False)

			# Degeneracy (2J + 1)
			degeneracies = 2 * J_num + 1

			# Expand energies: level → states
			energies_states = np.repeat(energies_levels, degeneracies)

			thermo_data_free = compute_thermodynamics_from_spectrum(energies_states, temperature_list, unit_want, degeneracies=None)

			# Extract heat capacity
			cv_values_free = [thermo_data_free[round(float(T), 1)]["heat_capacity"] for T in temperature_list]

			# Cumulative population at T = 100 K
			T_target = 100.0
			cum_populations_free = thermo_data_free[T_target]["cum_populations"]

			# State index (not J anymore)
			states_free = np.arange(1, len(cum_populations_free) + 1)

		for curve_idx, ((jmax, E), thermo_data) in enumerate(thermo_dict.items()):
			cv_values = [thermo_data[round(T, 1)]["heat_capacity"] for T in temperature_list]
			unit_cv = thermo_data[temperature_list[0]]["display_cv_unit"]

			cum_populations_field = thermo_data[T_target]["cum_populations"]
			states_field = np.arange(1, len(cum_populations_field) + 1)

			if num_molecules > 1:
				ax.plot(
					temperature_list,
					cv_values,
					color=color_cycle[mol_idx],
					linestyle=line_styles[mol_idx],
					linewidth=1.5,
					label=molecule,
				)

				# Plot the classical limit only once
				if mol_idx == num_molecules - 1:
					ax.plot(
						temperature_list,
						cv_array,
						color="black",
						linestyle=(0, (3, 1, 1, 1, 1, 1)),
						linewidth=1.5,
						label=r"Classical limit ($C_V = k_\mathrm{B}$)",
					)


			if num_molecules == 1:
				axs[0].plot(
					temperature_list,
					cv_values,
					color=color_cycle[0],
					linestyle=line_styles[mol_idx],
					linewidth=1.5,
					label=rf"{molecule} ($E={E:.0f}\,\mathrm{{kV/cm}}$)"
				)
 
				axs[0].plot(
					temperature_list,
					cv_values_free,
					color=color_cycle[1],
					linestyle=line_styles[mol_idx+1],
					linewidth=1.5,
					label=rf"{molecule} (Field-free)"
				)
 
				# Plot
				axs[0].plot(
					temperature_list,
					cv_array,
					color="black",
					linestyle=(0, (3, 1, 1, 1, 1, 1)),
					linewidth=1.5,
					label=r"Classical limit ($C_V = k_\mathrm{B}$)",
				)


				# Static electric field
				axs[1].plot(
					states_field,
					cum_populations_field,
					linestyle='none',
					marker='o',
					markersize=7,
					markerfacecolor=color_cycle[0],
					markeredgecolor=color_cycle[0],
					alpha=1.0,
					label=rf"{molecule} ($E={E:.0f}\,\mathrm{{kV/cm}}$)"
				)

				# Field-free rotor
				axs[1].plot(
					states_free,
					cum_populations_free,
					linestyle='none',
					marker='s',
					markersize=7,
					markerfacecolor='none',
					markeredgecolor=color_cycle[1],
					markeredgewidth=1.5,
					alpha=0.6,
					label=rf"{molecule} (Field-free)"
				)

	if num_molecules == 1:

		safe_unit = unit_cv.replace("^-1", "$^{-1}$")

		# -----------------------------
		# Heat-capacity panel
		# -----------------------------
		axs[0].set_xlabel("Temperature (K)")
		axs[0].set_ylabel(rf"$C_V$ [{safe_unit}]")
		axs[0].set_xlim(-2.0, 102)
		axs[0].set_ylim(-0.01, 0.801)
		axs[0].minorticks_on()
		axs[0].legend(loc="best")

		# -----------------------------
		# Cumulative population panel
		# -----------------------------
		axs[1].set_xlabel("Eigenstate index (in ascending energy)")
		axs[1].set_ylabel("Cumulative Boltzmann population")
		axs[1].set_ylim(-0.01, 1.01)
		axs[1].minorticks_on()
		axs[1].legend(loc="lower right")

		# -----------------------------
		# Common formatting
		# -----------------------------
		for i, ax in enumerate(axs):
			ax.margins(x=0.02)
			ax.xaxis.set_minor_locator(AutoMinorLocator())
			ax.yaxis.set_minor_locator(AutoMinorLocator())

			# Panel labels
			ax.text(
				-0.10, 1.03,
				f"({chr(97+i)})",
				transform=ax.transAxes,
				va="bottom",
				ha="left",
			)

	if num_molecules != 1:

		safe_unit = unit_cv.replace("^-1", "$^{-1}$")

		# -----------------------------
		# Axis labels
		# -----------------------------
		ax.set_xlabel("Temperature (K)")
		ax.set_ylabel(rf"$C_V$ [{safe_unit}]")

		# -----------------------------
		# Axis limits
		# -----------------------------
		ax.set_xlim(-2.0, 102)
		ax.set_ylim(-0.01, 0.801)

		# -----------------------------
		# Tick locations
		# -----------------------------
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())

		# -----------------------------
		# Legend
		# -----------------------------
		ax.legend(loc="best")
	# -----------------------------
	# Layout
	# -----------------------------
	plt.tight_layout()

	# Save first, then show
	plt.savefig(out_path, dpi=300)
	print("")
	print(f"[INFO] Combined Cv plot saved: {out_path}")

def compute_thermodynamics_from_spectrum(eigenvalues, temperature_list, unit, degeneracies=None, pop_tol=1e-16, cum_tol=1 - 1e-14):
	"""
	Compute thermodynamic properties from a given energy spectrum.

	Parameters
	----------
	eigenvalues : array_like
		Energies (cm^-1). Can be level-resolved or state-resolved.
	temperature_list : array_like
		Temperatures in Kelvin.
	unit : {'wavenumber', 'SI'}
		Output unit system.
	degeneracies : array_like or None, optional
		Degeneracy for each eigenvalue. If None → state-resolved (g = 1).
	pop_tol : float
		Absolute Boltzmann weight cutoff.
	cum_tol : float
		Cumulative population threshold.

	Returns
	-------
	dict
		Thermodynamic quantities indexed by temperature.
	"""

	energies = np.asarray(eigenvalues, dtype=np.float64)
	if energies.ndim != 1:
		raise ValueError("Eigenvalues must be 1D.")

	if unit not in {"wavenumber", "SI"}:
		raise ValueError("Invalid unit.")

	# Degeneracy handling
	if degeneracies is None:
		g = np.ones_like(energies)
	else:
		g = np.asarray(degeneracies, dtype=np.float64)
		if g.shape != energies.shape:
			raise ValueError("Degeneracies must match eigenvalues shape.")

	# Sort energies
	sort_idx = np.argsort(energies)
	energies = energies[sort_idx]
	g = g[sort_idx]

	# Constants
	kB = 0.69503476  # cm^-1/K

	# Energy shift (numerical stability)
	E0 = energies[0]
	Delta = energies - E0

	results = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"T must be > 0. Got {T}")

		beta = 1.0 / (kB * T)

		# ---- Full Boltzmann weights ----
		weights = g * np.exp(-beta * Delta)

		if not np.any(weights > 0):
			raise RuntimeError(f"All Boltzmann weights underflow at T={T}")

		# ---- Partition function (FULL) ----
		Z = np.sum(weights)

		# ---- Full normalized probabilities ----
		populations_full = weights / Z

		# ==========================================================
		# Convergence check (DO NOT use for observables)
		# ==========================================================
		mask = weights > pop_tol
		populations_check = populations_full[mask]
		cum_pop = np.cumsum(populations_check)

		weights_mask = weights[mask]
		Z_mask = np.sum(weights_mask)

		missing_pop = 1.0 - (Z_mask / Z)

		if missing_pop > (1.0 - cum_tol):
			raise RuntimeError(
				f"Population convergence NOT reached at T={T} K.\n"
				f"Missing population = {missing_pop:.6e} exceeds tolerance {1.0 - cum_tol:.6e}.\n"
				f"Increase basis size or relax tolerances."
			)
		# ==========================================================
		# Observables (ALWAYS FULL SPACE)
		# ==========================================================

		# ---- Energy moments ----
		E_avg = np.dot(populations_full, energies)
		E2_avg = np.dot(populations_full, energies**2)

		Cv_cm1 = kB * beta**2 * (E2_avg - E_avg**2)

		# ---- Adaptive truncation (dominant criterion) ----
		idx_conv = len(cum_pop)

		# ---- Unit conversion ----
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

		T_key = round(float(T), 1)
		results[T_key] = {
			"temperature_K": T,
			"beta": beta,
			"partition_function": Z,
			"populations_full": populations_full,
			"populations_check": populations_check,
			"cum_populations": cum_pop,
			"internal_energy": U_out,
			"heat_capacity": Cv_out,
			"convergence_index": idx_conv,
			"convergence_energy": energies[idx_conv],
			"unit": unit,
			"display_unit": display_unit,
			"display_cv_unit": display_cv_unit
		}

	return results

def compute_thermo_vectorized(JM_list, eigenvalues, eigenvectors, temperature_list, unit):
	"""
	Compute thermodynamic properties (Z, populations, cumulative-population truncation, U, Cv) from energy eigenvalues,
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

	pop_tol : float, optional (default=1e-10)
	Absolute cutoff for Boltzmann weights. States with weights below this
	threshold are discarded as numerically insignificant.

	cum_tol : float, optional (default=1 - 1e-10)
	Cumulative population threshold used for adaptive truncation. The summation
	over states is truncated once the cumulative Boltzmann population reaches
	this value, ensuring that neglected states contribute negligibly.

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
			- cum_populations	: Cumulative sum of populations, used for convergence assessment.
			- internal_energy	: Mean energy (U)
			- heat_capacity		: Heat capacity (Cv)
			- levels_used		: Number of energy levels included
			- convergence_energy: Energy at which convergence was met (in cm⁻¹)
			- convergence_index : Index where threshold was first met
	"""
	# --- Precompute once ---
	i_idx, j_idx, A = precompute_coupling_arrays(JM_list)

	#print("\n\n")
	#for k in range(10):
	#	print(f"{k}: i={i_idx[k]}, j={j_idx[k]}, A={A[k]:.6f}")
	#
	#print("\n\n")

	# --- Extract coupled components ---
	C_i = eigenvectors[i_idx, :]   # (n_pairs, N_states)
	C_j = eigenvectors[j_idx, :]   # (n_pairs, N_states)

	# --- Complex-safe overlap ---
	overlaps = np.real(np.conj(C_i) * C_j)

	# --- Sum over basis pairs ---
	pair_sum = np.sum(A[:, None] * overlaps, axis=0)  # (N_states,)

	energies = np.asarray(eigenvalues, dtype=np.float64)
	if energies.ndim != 1:
		raise ValueError("Eigenvalues must be a one-dimensional array.")
	if unit not in {"wavenumber", "SI"}:
		raise ValueError("Unit must be either 'wavenumber' or 'SI'.")

	kb = KB_CM_INV_PER_K  # cm^-1/K

	results = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"Temperature must be > 0 K. Got: {T}")

		(
			ground_state_energy,
			beta,
			boltzmann_weights,
			partition_function_shifted,
			probabilities,
			cum_pop,
		) = compute_boltzmann_probabilities(
			eigenvalues=eigenvalues,
			temperature=T,
		)

		# ==========================================================
		# Observables (ALWAYS FULL SPACE)
		# ==========================================================

		# ---- Energy moments ----
		E_avg = np.dot(probabilities, energies)
		E2_avg = np.dot(probabilities, energies**2)

		Cv_cm1 = kb * beta**2 * (E2_avg - E_avg**2)

		# ---- Orientation ----
		#total = np.sum(weights * pair_sum, axis=1)
		total = np.dot(boltzmann_weights, pair_sum)   # FULL
		cos_theta_avg = (2.0 / partition_function_shifted) * total

		# ---- Adaptive truncation (dominant criterion) ----
		idx_conv = len(cum_pop)

		# ---- Unit conversion ----
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

		# --------------------------------------------------------
		# Compute angular distribution
		# --------------------------------------------------------

		angular_distribution_results = (
			compute_angular_distribution_from_eigensystem(
				eigenvalues=eigenvalues,
				eigenvectors=eigenvectors,
				basis=JM_list,
				temperature=T,
				probabilities=probabilities,
				n_quad=101,
			)
		)

		x = angular_distribution_results["x"]
		w = angular_distribution_results["weights"]
		P_x = angular_distribution_results["P_x"]
		
		print(f"\n{'Quadrature Information':^50}")
		print("=" * 50)
		print(f"{'Number of quadrature points':<30}: {len(x)}")
		print("-" * 50)

		print(f"{'Index':>8} {'Quadrature Point':>20} {'Weight':>20}")
		print("-" * 50)

		for i, (xi, wi) in enumerate(zip(x, w), start=1):
			print(f"{i:8d} {xi:20.15f} {wi:20.15f}")

		print("=" * 50)


		print(f"Tr(rho) = " f"{angular_distribution_results['trace_rho'].real:.15f}")
		print(f"Integral P(x) dx = " f"{angular_distribution_results['normalization']:.15f}")
		whoami()

		T_key = round(float(T), 1)
		results[T_key] = {
			"temperature_K": T,
			"beta": beta,
			"partition_function": partition_function_shifted,
			"populations_full": probabilities,
			"cum_populations": cum_pop,
			"internal_energy": U_out,
			"heat_capacity": Cv_out,
			"dipole_orientation": cos_theta_avg,
			"convergence_index": idx_conv,
			"convergence_energy": energies[idx_conv],
			"unit": unit,
			"display_unit": display_unit,
			"display_cv_unit": display_cv_unit
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

				# --- Load quantum numbers ---
				JM_list_var = nc.variables["quantum_numbers_for_spin_state"]
				JM_list = np.array(JM_list_var[:, :], dtype=int)   # (N, 2)

				# --- Load eigenvalues ---
				eigenval_var = nc.variables["eigenvalues"]
				eigenvalues = np.array(eigenval_var[:], dtype=float)  # (N,)

				# --- Load eigenvectors (real + imaginary) ---
				real_var = nc.variables["real_eigenvectors"]
				imag_var = nc.variables["imag_eigenvectors"]

				real_evecs = np.array(real_var[:, :], dtype=float)
				imag_evecs = np.array(imag_var[:, :], dtype=float)

				# --- Construct complex eigenvectors ---
				eigenvectors = real_evecs + 1j * imag_evecs   

				# --- Metadata ---
				unit_from_file = getattr(eigenval_var, "units", "unknown")
				label_from_file = getattr(eigenval_var, "long_name", "eigenvalues")

				# --- Diagnostics ---
				print(f"[INFO] {'Eigenstates':<15}: {len(eigenvalues)}")
				print(f"[INFO] {'Units':<15}: {unit_from_file}")
				print(f"[INFO] {'Description':<15}: {label_from_file}")

				print(f"[INFO] {'JM_list shape':<15}: {JM_list.shape}")
				print(f"[INFO] {'Eigenvec shape':<15}: {eigenvectors.shape}")
				print(f"[INFO] {'Data type':<15}: {eigenvectors.dtype}")


				norms = np.sum(np.abs(eigenvectors)**2, axis=0)

				if not np.allclose(norms, 1.0, atol=1e-6):
					print("[WARNING] Eigenvectors not normalized → trying transpose")
					eigenvectors = eigenvectors.T

					norms = np.sum(np.abs(eigenvectors)**2, axis=0)
					if not np.allclose(norms, 1.0, atol=1e-6):
						raise ValueError("Eigenvectors are not orthonormal.")


				thermo_data = compute_thermo_vectorized(
					JM_list,
					eigenvalues=eigenvalues,
					eigenvectors=eigenvectors,
					temperature_list=temperature_list,
					unit=unit_want
				)

				# Print summary
				print("\n[INFO] Thermodynamic Summary:")
				for T in temperature_list:
					T_key = round(float(T), 1)
					entry = thermo_data[T_key]
					print(f"\n[ ] {'T':<30}= {T} K")
					#print(f"[ ] {'convergence_energy':<30}= {entry['convergence_energy']} {entry['display_unit']}")
					 # Convergence energy with conditional unit display
					convergence_energy = entry.get("convergence_energy")
					display_unit = entry.get("display_unit", "")
					if isinstance(convergence_energy, (int, float)) and not math.isnan(convergence_energy):
						print(f"[ ] {'convergence_energy':<30}= {convergence_energy:.6f} {display_unit}")
					else:
						print(f"[ ] {'convergence_energy':<30}= N/A")
					print(f"[ ] {'convergence size':<30}= {entry['convergence_index']}")
					print(f"[ ] {'final cumalative population':<30}= {(entry['cum_populations'][-1])}")
					print(f"[ ] {'Z':<30}= {entry['partition_function']:.6f}")
					print(f"[ ] {'U':<30}= {entry['internal_energy']:.6f} {entry['display_unit']}")
					print(f"[ ] {'Cv':<30}= {entry['heat_capacity']:.6f} {entry['display_cv_unit']}")
					print(f"[ ] {'<cosθ>':<30}= {entry['dipole_orientation']:.6f}\n")
				
				print("\n\n")

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

def precompute_coupling_arrays(JM_list):
	JM_to_index = {tuple(jm): i for i, jm in enumerate(JM_list)}
	#for key, value in JM_to_index.items():
	#	print(f"(J, M) = {key}  →  index = {value}")

	i_list, j_list, A_list = [], [], []

	for i, (J, M) in enumerate(JM_list):
		key = (J + 1, M)
		if key in JM_to_index:
			j = JM_to_index[key]

			A = np.sqrt(((J + 1)**2 - M**2) /
						((2*J + 1) * (2*J + 3)))

			i_list.append(i)
			j_list.append(j)
			A_list.append(A)

	return (np.array(i_list),
			np.array(j_list),
			np.array(A_list))


def compute_cos_theta_vectorized(evals, evecs, i_idx, j_idx, A, T_list):
	"""
	Compute <cos(theta)>_T for multiple temperatures.

	evals : (N,)
	evecs : (N_basis, N_states)  complex
	i_idx, j_idx : coupling indices
	A : coupling coefficients
	T_list : array of temperatures
	"""

	kB = 0.69503476  # cm^-1/K
	T_array = np.array(T_list)
	beta = 1.0 / (kB * T_array)[:, None]   # (nT, 1)

	# --- Extract coupled components ---
	C_i = evecs[i_idx, :]   # (n_pairs, N_states)
	C_j = evecs[j_idx, :]   # (n_pairs, N_states)

	# --- Complex-safe overlap ---
	overlaps = np.real(np.conj(C_i) * C_j)

	# --- Sum over basis pairs ---
	pair_sum = np.sum(A[:, None] * overlaps, axis=0)  # (N_states,)

	# --- Boltzmann weights ---
	weights = np.exp(-beta * evals[None, :])  # (nT, N_states)

	Z = np.sum(weights, axis=1)
	total = np.sum(weights * pair_sum, axis=1)

	return T_array, (2.0 / Z) * total


def plot_dipole_panel(
	ax,
	E,
	thermo_dict_by_molecule,
	get_temperature_list,
	styles,
):
	"""Plot the thermal dipole orientation for a single electric field."""

	temperature_list = get_temperature_list(dipole_orientation=True,)
	for molecule, thermo_dict in thermo_dict_by_molecule.items():

		if ( len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple))):
			temperature_list = temperature_list[0]

		for (jmax, Ef), thermo_data in thermo_dict.items():

			if Ef != E:
				continue

			dipole = [
				thermo_data[T]["dipole_orientation"]
				for T in temperature_list
			]

			ax.plot(
				temperature_list,
				dipole,
				color="black",
				linestyle=styles[molecule]["linestyle"],
				label=molecule,
			)

	# Electric field annotation
	ax.text(
		0.50,
		0.90,
		fr"$E = {E:.0f}\,\mathrm{{kV/cm}}$",
		transform=ax.transAxes,
		ha="center",
		va="center",
		bbox=dict(
			boxstyle="round",
			facecolor="white",
			edgecolor="black",
		),
	)

	# Minor ticks
	#ax.minorticks_on()
	#ax.xaxis.set_minor_locator(AutoMinorLocator())
	#ax.yaxis.set_minor_locator(AutoMinorLocator())

	# Slightly thicker border than the default
	#for spine in ax.spines.values():
	#	spine.set_linewidth(1.5)


def add_panel_labels(axes):
	"""Add panel labels (a), (b), (c), ... to each subplot."""

	for i, ax in enumerate(axes):
		ax.text(
			0.03,
			0.97,
			f"({chr(97 + i)})",
			transform=ax.transAxes,
			ha="left",
			va="top",
		)


def set_common_ylim(axes, ymin=0.0, padding=0.05):
	"""Set identical y-limits for all subplots."""

	ymax = max(ax.get_ylim()[1] for ax in axes)

	for ax in axes:
		ax.set_ylim(ymin, ymax * (1.0 + padding))


def plot_dipole_orientation_comparison(
	thermo_dict_by_molecule,
	electric_field_list,
	get_temperature_list,
	figsize=(12, 9),
	save_path=None,
):
	"""
	Plot the thermal dipole orientation, <cos(theta)>, as a function of
	temperature for multiple electric fields in a 2×2 panel.

	Parameters
	----------
	thermo_dict_by_molecule : dict
		Thermodynamic results for all molecules.

	electric_field_list : sequence
		Electric field strengths (typically four).

	get_temperature_list : callable
		Function returning the temperature grid.

	styles : dict
		Plot style for each molecule.

	figsize : tuple, optional
		Figure size in inches.

	save_path : str or pathlib.Path, optional
		Output filename. If None, the figure is not saved.

	Returns
	-------
	fig : matplotlib.figure.Figure
	axes : ndarray of matplotlib.axes.Axes
	"""

	set_plot_style()

	fig, axes = plt.subplots(
		2,
		2,
		figsize=figsize,
		sharex=True,
		sharey=True,
	)

	axes = axes.ravel()

	# Plot each electric field
	for ax, E in zip(axes, electric_field_list):
		plot_dipole_panel(
			ax=ax,
			E=E,
			thermo_dict_by_molecule=thermo_dict_by_molecule,
			get_temperature_list=get_temperature_list,
			styles=styles,
		)


	for ax in axes:
		ax.margins(x=0.02)

	add_panel_labels(axes)
	set_common_ylim(axes)

	# Common axis labels
	axes[0].set_ylabel(r"$\langle\cos\theta\rangle_T$")
	axes[2].set_ylabel(r"$\langle\cos\theta\rangle_T$")

	axes[2].set_xlabel(r"Temperature (K)")
	axes[3].set_xlabel(r"Temperature (K)")

	# Common legend
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		loc="upper center",
		ncol=len(labels),
		bbox_to_anchor=(0.5, 1.02),
	)

	fig.tight_layout(rect=(0, 0, 1, 0.95))

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)

		fig.savefig(save_path)

		print("=" * 80)
		print(f"[ ] Figure saved: {save_path.resolve()}")
		print("=" * 80)

	return fig, axes


def get_ground_state_dipole_orientation(thermo_dict_by_molecule, get_temperature_list):
	output = []

	temperature_list = get_temperature_list(dipole_orientation=True)
	for molecule, thermo_dict in thermo_dict_by_molecule.items():

		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		T_min = min(temperature_list)

		output.append(f"\nMolecule: {molecule} (using T = {T_min} K)")

		B_const = MOLECULE_DATA[molecule]["B_const"]
		dipole_moment = MOLECULE_DATA[molecule]["dipole_moment"]

		for (jmax, E), thermo_data in thermo_dict.items():

			if T_min not in thermo_data:
				raise KeyError(
					f"T={T_min} missing for {molecule}, jmax={jmax}, E={E}"
				)

			val_num = thermo_data[T_min]["dipole_orientation"]

			potential_strength = convert_dipole_field_energy_to_cm_inv(
				dipole_moment, E
			)

			x = potential_strength / B_const

			val_ana_norm = (x / 3.0) * (1.0 - x**2 / 12.0)
			error_norm = abs(val_num - val_ana_norm)
			Z = thermo_data[T_min]["partition_function"]

			output.append(
				f"{molecule:4s}  "
				f"jmax={jmax:2d}  "
				f"E={E:8.1f}  "
				f"x={x:8.4f}  "
				f"Z={Z:10.4f}  "
				f"<cosθ>_num={val_num:.6f}  "
				f"<cosθ>_ana={val_ana_norm:.6f}  "
				f"Δ={error_norm:.2e}"
			)

	# Executed once after ALL molecules have been processed
	print("\n".join(output))


# ============================================================
# SPHERICAL-HARMONIC NORMALIZATION
# ============================================================

def spherical_harmonic_normalization(J, M):
	r"""
	Compute the normalization constant

		N_J^M =
		sqrt[
			(2J+1)/(4*pi)
			(J-|M|)!/(J+|M|)!
		].

	Notes
	-----
	scipy.special.lpmv(m, J, x) includes the Condon--Shortley
	phase (-1)^m. Therefore, no additional (-1)^M factor is
	included here.
	"""

	m = abs(M)

	log_norm_squared = (
		np.log(2.0 * J + 1.0)
		- np.log(4.0 * np.pi)
		+ gammaln(J - m + 1.0)
		- gammaln(J + m + 1.0)
	)

	return np.exp(0.5 * log_norm_squared)


# ============================================================
# THERMAL BOLTZMANN PROBABILITIES
# ============================================================

def compute_boltzmann_probabilities(
	eigenvalues,
	temperature,
	kb=KB_CM_INV_PER_K,
	pop_tol=1e-16, 
	cum_tol=1-1e-14
):
	r"""
	Compute canonical Boltzmann probabilities.

	The canonical probability of state n is

		p_n = exp(-beta E_n) / Z,

	where

		beta = 1 / (k_B T).

	For numerical stability, the energies are shifted by the
	ground-state energy before evaluating the Boltzmann factors.

	A population-convergence check is performed using ``pop_tol``
	and ``cum_tol``. The convergence check is diagnostic only and
	does not modify the probabilities returned for observables.

	Parameters
	----------
	eigenvalues : ndarray, shape (n_states,)
		Energy eigenvalues in cm^{-1}.

	temperature : float
		Temperature in K. Must be greater than zero.

	kb : float, optional
		Boltzmann constant in cm^{-1} K^{-1}.

	pop_tol : float, optional
		Threshold below which a Boltzmann weight is considered
		negligible for the population-convergence check.

	cum_tol : float, optional
		Required cumulative population tolerance. The retained
		population must be at least ``cum_tol``.

	Returns
	-------
	probabilities : ndarray
		Normalized Boltzmann probabilities for all eigenstates.

	beta : float
		Inverse temperature in (cm^{-1})^{-1}.

	partition_function_shifted : float
		Partition function evaluated using energies shifted by
		the ground-state energy.

	ground_state_energy : float
		Ground-state energy, i.e. the minimum eigenvalue.

	Raises
	------
	ValueError
		If ``temperature`` is not positive, or if ``pop_tol`` or
		``cum_tol`` is outside its valid range.

	RuntimeError
		If the population-convergence criterion is not satisfied.
	"""

	# ==========================================================
	# Input validation
	# ==========================================================
	eigenvalues = np.asarray(
		eigenvalues,
		dtype=np.float64,
	)

	if eigenvalues.ndim != 1:
		raise ValueError(
			"eigenvalues must be a one-dimensional array."
		)

	if eigenvalues.size == 0:
		raise ValueError(
			"eigenvalues must not be empty."
		)

	if temperature <= 0.0:
		raise ValueError(
			"Temperature must be greater than zero."
		)

	if pop_tol <= 0.0:
		raise ValueError(
			"pop_tol must be greater than zero."
		)

	if not 0.0 < cum_tol <= 1.0:
		raise ValueError(
			"cum_tol must satisfy 0 < cum_tol <= 1."
		)

	# ==========================================================
	# Inverse temperature
	# ==========================================================
	beta = 1.0 / (kb * temperature)

	# ==========================================================
	# Shift energies by the ground-state energy
	# ==========================================================
	ground_state_energy = np.min(eigenvalues)
	shifted_energies = (eigenvalues - ground_state_energy)

	# ==========================================================
	# Boltzmann weights
	# ==========================================================
	boltzmann_weights = np.exp(-beta * shifted_energies)

	# ==========================================================
	# Shifted partition function
	# ==========================================================
	partition_function_shifted = np.sum(boltzmann_weights)

	# ==========================================================
	# Normalized Boltzmann probabilities
	#
	# These probabilities are computed from ALL states and
	# must be used for physical observables.
	# ==========================================================
	probabilities = (boltzmann_weights / partition_function_shifted)

	# ==========================================================
	# Population convergence check
	#
	# This check does NOT truncate the probabilities used for
	# observables.
	# ==========================================================

	mask = boltzmann_weights > pop_tol
	boltzmann_weights_mask = (boltzmann_weights[mask])
	partition_function_mask = np.sum(boltzmann_weights_mask)
	retained_population = (partition_function_mask / partition_function_shifted)

	missing_population = (1.0 - retained_population)

	if missing_population > (1.0 - cum_tol):
		raise RuntimeError(
			f"Population convergence NOT reached "
			f"at T={temperature} K.\n"
			f"Missing population = {missing_population:.6e} "
			f"exceeds tolerance {1.0 - cum_tol:.6e}.\n"
			f"Increase basis size or relax tolerances."
		)

	probabilities_mask = (probabilities[mask])
	cum_pop = np.cumsum(probabilities_mask)

	return (
		ground_state_energy,
		beta,
		boltzmann_weights,
		partition_function_shifted,
		probabilities,
		cum_pop,
	)

# ============================================================
# EXTRACT M BLOCKS OF THE DENSITY MATRIX
# ============================================================

def extract_density_matrix_M_blocks(
	rho,
	basis,
):
	r"""
	Extract

		rho_{J,J'}^(M)
		=
		<J,M | rho | J',M>

	from the full density matrix.

	Parameters
	----------
	rho : ndarray, shape (n_basis, n_basis)
		Full thermal density matrix.

	basis : list of tuple
		Basis ordering:

			basis[i] = (J, M)

	Returns
	-------
	rho_by_M : dict
		Dictionary with

			rho_by_M[M]["J_values"]
			rho_by_M[M]["rho"]
	"""

	basis = list(basis)
	#for i, (J, M) in enumerate(basis[:20]):
	#	print(f"{i:3d}: J = {J:3d}, M = {M:3d}")

	#print("\n\n")

	if rho.shape[0] != len(basis):
		raise ValueError(
			"The dimension of rho does not match the basis size."
		)

	# Unique M values
	M_values = sorted(
		set(M for J, M in basis)
	)

	#for i, M in enumerate(M_values[:20]):
	#	print(f"{i:3d}: M = {M:3d}")

	rho_by_M = {}

	for M in M_values:

		# Indices belonging to this M sector
		indices = [
			i
			for i, (J, M_basis) in enumerate(basis)
			if M_basis == M
		]

		#print(f"M = {M:3d} : indices = {indices}")

		# Sort by J
		indices = sorted(
			indices,
			key=lambda i: basis[i][0],
		)

		J_values = np.array(
			[
				basis[i][0]
				for i in indices
			],
			dtype=int,
		)

		# Extract M block
		rho_M = rho[
			np.ix_(
				indices,
				indices,
			)
		]

		rho_by_M[M] = {
			"J_values": J_values,
			"rho": rho_M,
		}

	return rho_by_M


# ============================================================
# COMPUTE P(x), x = cos(theta)
# ============================================================

def compute_angular_probability_density(
	rho_by_M,
	n_quad=300,
):
	r"""
	Compute

		P(x)
		=
		2*pi
		sum_M
		sum_{J,J'}
		rho_{J,J'}^(M)
		N_J^M
		N_{J'}^M
		P_J^M(x)
		P_{J'}^M(x),

	where

		x = cos(theta).

	Gauss--Legendre quadrature is used.

	Parameters
	----------
	rho_by_M : dict
		Density-matrix blocks returned by
		extract_density_matrix_M_blocks().

	n_quad : int
		Number of Gauss--Legendre quadrature points.

	Returns
	-------
	x : ndarray
		Gauss--Legendre nodes in [-1,1].

	weights : ndarray
		Gauss--Legendre quadrature weights.

	P_x : ndarray
		Probability density P(x).

	normalization : float
		Numerical value of

			integral_{-1}^{1} P(x) dx.
	"""

	# Gauss--Legendre nodes and weights
	x, weights = leggauss(n_quad)

	P_x = np.zeros(
		n_quad,
		dtype=np.complex128,
	)

	# --------------------------------------------------------
	# Sum over M sectors
	# --------------------------------------------------------

	for M, data in rho_by_M.items():

		J_values = data["J_values"]
		rho_M = data["rho"]

		n_J = len(J_values)

		# ----------------------------------------------------
		# Phi[J_index, x_index]
		#
		# =
		#
		# N_J^M P_J^M(x)
		# ----------------------------------------------------

		Phi = np.empty(
			(n_J, n_quad),
			dtype=np.float64,
		)

		m = abs(M)

		for j_index, J in enumerate(J_values):

			N_JM = (
				spherical_harmonic_normalization(
					J=J,
					M=M,
				)
			)

			P_JM = lpmv(
				m,
				J,
				x,
			)

			Phi[j_index, :] = (
				N_JM
				* P_JM
			)

		# ----------------------------------------------------
		# At each x_i:
		#
		# P_M(x_i)
		# =
		# 2*pi
		# sum_{J,J'}
		# rho_{J,J'}^(M)
		# Phi_J^M(x_i)
		# Phi_{J'}^M(x_i)
		# ----------------------------------------------------

		P_x += (
			2.0
			* np.pi
			* np.einsum(
				"ji,jk,ki->i",
				Phi.conj(),
				rho_M,
				Phi,
				optimize=True,
			)
		)

	# The result must be real
	max_imaginary_part = np.max(
		np.abs(P_x.imag)
	)

	if max_imaginary_part > 1.0e-10:
		print(
			"Warning: P(x) contains a non-negligible "
			f"imaginary part: {max_imaginary_part:.3e}"
		)

	P_x = np.real(P_x)

	# Gauss--Legendre normalization
	normalization = np.sum(
		weights * P_x
	)

	return (
		x,
		weights,
		P_x,
		normalization,
	)



# ============================================================
# MASTER FUNCTION
# ============================================================

def compute_angular_distribution_from_eigensystem(
	basis,
	eigenvalues,
	eigenvectors,
	temperature,
	probabilities,
	n_quad=300,
	kb=KB_CM_INV_PER_K,
):
	"""
	Compute the thermal angular probability distribution
	directly from supplied eigenvalues and eigenvectors.

	Parameters
	----------
	eigenvalues : ndarray, shape (n_states,)
		Eigenvalues in cm^{-1}.

	eigenvectors : ndarray, shape (n_basis, n_states)
		Eigenvectors. Column n is the nth eigenvector.

	basis : list of tuple
		Basis ordering:

			[(J_0, M_0), (J_1, M_1), ...]

	temperature : float
		Temperature in K.

	n_quad : int
		Number of Gauss--Legendre quadrature points.

	Returns
	-------
	results : dict
		Dictionary containing all computed quantities.
	"""

	# --------------------------------------------------------
	# Thermal density matrix
	# --------------------------------------------------------

	rho = (
		eigenvectors
		@ np.diag(probabilities)
		@ eigenvectors.conj().T
	)

	# --------------------------------------------------------
	# Extract fixed-M blocks
	# --------------------------------------------------------

	rho_by_M = (
		extract_density_matrix_M_blocks(
			rho=rho,
			basis=basis,
		)
	)

	# --------------------------------------------------------
	# Compute P(x)
	# --------------------------------------------------------

	(
		x,
		weights,
		P_x,
		normalization,
	) = compute_angular_probability_density(
		rho_by_M=rho_by_M,
		n_quad=n_quad,
	)

	# --------------------------------------------------------
	# Trace of density matrix
	# --------------------------------------------------------

	trace_rho = np.trace(rho)

	return {
		"x": x,
		"weights": weights,
		"P_x": P_x,
		"rho": rho,
		"rho_by_M": rho_by_M,
		"normalization": normalization,
		"trace_rho": trace_rho,
	}
