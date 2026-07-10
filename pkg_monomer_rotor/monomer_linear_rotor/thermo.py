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
from scipy.constants import R as GAS_CONSTANT_J_PER_MOL_K
from scipy.constants import k as BOLTZMANN_J_PER_K
import warnings
import itertools
from termcolor import colored

from monomer_linear_rotor.molecule_data import MOLECULE_DATA

from monomer_linear_rotor.utils import convert_dipole_field_energy_to_cm_inv


from monomer_linear_rotor.utils import (
	wavenumber_to_joules_per_mole,
)

from monomer_linear_rotor.hamiltonian import (
	rotational_energy_levels,
)

from pkg_utils.utils import whoami
from pkg_utils.config import *
from pkg_utils.env_report import whom


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
	num_molecules = len(thermo_dict_by_molecule)	
	# -----------------------------
	# Figure setup
	# -----------------------------
	if num_molecules != 1:
		fig, ax = plt.subplots(figsize=(9, 6))

	else:
		# Create figure with 2 row, 1 columns
		fig, axs = plt.subplots(2, 1, figsize=(9, 12))

	color_cycle = ["red", "grey", "blue", "green"]
	line_styles = ["-", "--", "-.", ":"]
	markers = ["o", "s", "p", "d", "v"]

	# Rotational heat capacity (constant)
	Cv_rot_equipartition_theorem = 0.695  # cm^-1 K^-1

	# Temperature range (K)
	T = np.linspace(0, 1000, 200)

	# Create an array of same size
	Cv_array = np.full_like(T, Cv_rot_equipartition_theorem)

	for mol_idx, (molecule, thermo_dict) in enumerate(thermo_dict_by_molecule.items()):
		temperature_list = get_temperature_list(molecule)

		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		color = color_cycle[mol_idx % len(color_cycle)]
		mk = markers[mol_idx % len(markers)]

		if num_molecules == 1:
			J_num, energies_free = rotational_energy_levels(MOLECULE_DATA[molecule]["B_const"], 2000)
			thermo_data_free = compute_thermo_vectorized_free(energies_free, temperature_list, unit_want, pop_tol=1e-10, cum_tol=1-1e-10)
			cv_values_free = [thermo_data_free[T]["heat_capacity"] for T in temperature_list]
			cum_populations_free = thermo_data_free[100]["cum_populations"]
			states_free = np.arange(1, len(cum_populations_free) + 1)


		for curve_idx, ((jmax, E), thermo_data) in enumerate(thermo_dict.items()):
			cv_values = [thermo_data[T]["heat_capacity"] for T in temperature_list]
			unit_cv = thermo_data[temperature_list[0]]["display_cv_unit"]

			cum_populations_field = thermo_data[100]["cum_populations"]
			states_field = np.arange(1, len(cum_populations_field) + 1)


			if num_molecules != 1:
				ls = line_styles[curve_idx % len(line_styles)]
				plt.plot(
					temperature_list,
					cv_values,
					linestyle='none',
					marker=mk,
					markersize=8,
					#`markerfacecolor='none',
					color=color,
					alpha=0.65,
					label=rf"{molecule}"
				)

				if ((mol_idx%num_molecules) == (num_molecules-1)):
					# Plot
					plt.plot(
						T,
						Cv_array,
						'k--',
						linewidth=2,
						label=r'Classical limit: $C_{V,\mathrm{rot}} = k_\mathrm{B} \approx 0.695\ \mathrm{cm^{-1}\,K^{-1}}$'
					)

			if num_molecules == 1:
				axs[0].plot(
					temperature_list,
					cv_values,
					linestyle='none',
					#linewidth=1.6,
					marker='o',
					markersize=8,
					#markerfacecolor='none',
					color=color,
					label=rf"{molecule} (static electric field, $E={E:.0f}\,\mathrm{{kV/cm}}$)"
				)

				axs[0].plot(
					temperature_list,
					cv_values_free,
					linestyle='none',
					#linewidth=1.4,
					marker='p',
					markersize=8,
					color="blue",
					alpha=0.65,
					label=rf"{molecule} (field-free rotor)"
				)

				# Plot
				axs[0].plot(
					T,
					Cv_array,
					'k--',
					linewidth=2,
					label=r'Classical limit: $C_{V,\mathrm{rot}} = R \approx 0.695\ \mathrm{cm^{-1}\,K^{-1}}$'
				)

				axs[1].plot(
					states_field,
					cum_populations_field,
					linestyle='none',
					#linewidth=1.6,
					marker='o',
					markersize=8,
					#markerfacecolor='none',
					color=color,
					label=rf"{molecule} (static electric field, $E={E:.0f}\,\mathrm{{kV/cm}}$)"
				)

				axs[1].plot(
					states_free,
					cum_populations_free,
					linestyle='none',
					#linewidth=1.4,
					marker='p',
					markersize=8,
					color="blue",
					alpha=0.65,
					label=rf"{molecule} (field-free rotor)"
				)

	if num_molecules == 1:
		# -----------------------------
		# Axis labels
		# -----------------------------
		axs[0].set_xlabel("Temperature (K)", fontsize=18)

		safe_unit = unit_cv.replace("^-1", "$^{-1}$")
		axs[0].set_ylabel(rf"Heat Capacity [{safe_unit}]", fontsize=18)

		# -----------------------------
		# Limits
		# -----------------------------
		axs[0].set_xlim(-0.5, 100.5)
		axs[0].set_ylim(-0.01, 0.8)

		# -----------------------------
		# Major ticks
		# -----------------------------
		axs[0].set_xticks(np.arange(0, 101, 10))
		axs[0].yaxis.set_major_locator(MultipleLocator(0.1))

		# -----------------------------
		# Minor ticks
		# -----------------------------
		axs[0].xaxis.set_minor_locator(MultipleLocator(2))
		axs[0].yaxis.set_minor_locator(MultipleLocator(0.02))

		# -----------------------------
		# Tick styling (publication quality)
		# -----------------------------
		axs[0].tick_params(axis='both', which='major',
					   direction='in', length=7, width=1.2,
					   labelsize=18, top=True, right=True)

		axs[0].tick_params(axis='both', which='minor',
					   direction='in', length=4, width=1.0,
					   top=True, right=True)

		# -----------------------------
		# Spines (frame thickness)
		# -----------------------------
		for spine in axs[0].spines.values():
			spine.set_linewidth(1.2)

		# -----------------------------
		# Optional: light grid (very subtle)
		# -----------------------------
		#ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.5)
		#ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

		axs[0].legend(fontsize=18, loc="best")
		# -----------------------------
		# -----------------------------
		# Axis labels
		# -----------------------------
		axs[1].set_xlabel("Index of states", fontsize=18)
		axs[1].set_ylabel(rf"Cumulative population", fontsize=18)

		# -----------------------------
		# Limits
		# -----------------------------
		#axs[1].set_xlim(-0.5, 100.5)
		axs[1].set_ylim(-0.01, 1.01)

		# -----------------------------
		# Major ticks
		# -----------------------------
		#axs[1].set_xticks(np.arange(0, 101, 10))
		#axs[1].yaxis.set_major_locator(MultipleLocator(0.1))

		# -----------------------------
		# Minor ticks
		# -----------------------------
		axs[1].xaxis.set_minor_locator(MultipleLocator(1))
		axs[1].yaxis.set_minor_locator(MultipleLocator(0.02))

		# -----------------------------
		# Tick styling (publication quality)
		# -----------------------------
		axs[1].tick_params(axis='both', which='major',
					   direction='in', length=7, width=1.2,
					   labelsize=18, top=True, right=True)

		axs[1].tick_params(axis='both', which='minor',
					   direction='in', length=4, width=1.0,
					   top=True, right=True)

		# -----------------------------
		# Spines (frame thickness)
		# -----------------------------
		for spine in axs[1].spines.values():
			spine.set_linewidth(1.2)

		# -----------------------------
		# Optional: light grid (very subtle)
		# -----------------------------
		#ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.5)
		#ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

		axs[1].legend(fontsize=18, loc="lower right")
		# -----------------------------

		# Labels (a), (b)
		labels = ["(a)", "(b)"]
		for i, ax in enumerate(axs):
			ax.text(0.02, 0.97, labels[i],
					transform=ax.transAxes,
					fontsize=20,
					fontweight='bold',
					va='top', ha='left')


	if num_molecules != 1:
		# -----------------------------
		# Axis labels
		# -----------------------------
		ax.set_xlabel("Temperature (K)", fontsize=18)

		safe_unit = unit_cv.replace("^-1", "$^{-1}$")
		ax.set_ylabel(rf"Heat Capacity [{safe_unit}]", fontsize=18)

		# -----------------------------
		# Limits
		# -----------------------------
		ax.set_xlim(-0.5, 100.5)
		ax.set_ylim(-0.01, 0.8)

		# -----------------------------
		# Major ticks
		# -----------------------------
		ax.set_xticks(np.arange(0, 101, 10))
		ax.yaxis.set_major_locator(MultipleLocator(0.1))

		# -----------------------------
		# Minor ticks
		# -----------------------------
		ax.xaxis.set_minor_locator(MultipleLocator(2))
		ax.yaxis.set_minor_locator(MultipleLocator(0.02))

		# -----------------------------
		# Tick styling (publication quality)
		# -----------------------------
		ax.tick_params(axis='both', which='major',
					   direction='in', length=7, width=1.2,
					   labelsize=18, top=True, right=True)

		ax.tick_params(axis='both', which='minor',
					   direction='in', length=4, width=1.0,
					   top=True, right=True)

		# -----------------------------
		# Spines (frame thickness)
		# -----------------------------
		for spine in ax.spines.values():
			spine.set_linewidth(1.2)

		# -----------------------------
		# Optional: light grid (very subtle)
		# -----------------------------
		#ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.5)
		#ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

		plt.legend(fontsize=18, loc="best")
		# -----------------------------
	# Layout
	# -----------------------------
	plt.tight_layout()

	# Save first, then show
	plt.savefig(out_path, dpi=300)
	print("")
	print(f"[INFO] Combined Cv plot saved: {out_path}")

def compute_thermo_vectorized_free(
	eigenvalues,
	temperature_list,
	unit,
	pop_tol=1e-10,
	cum_tol=1 - 1e-10
):
	"""
	Vectorized thermodynamics for a free linear rigid rotor including degeneracy.

	Parameters
	----------
	eigenvalues : array_like or dict
		Rotational energies (cm^-1). If dict, keys assumed to be J.
	temperature_list : array_like
		Temperatures in Kelvin.
	unit : {'wavenumber', 'SI'}
		Output unit system.
	pop_tol : float, optional
		Absolute Boltzmann weight cutoff.
	cum_tol : float, optional
		Cumulative population threshold.

	Returns
	-------
	dict
		Thermodynamic quantities indexed by temperature.
	"""

	# ---- Handle dict input (J → E_J) ----
	if isinstance(eigenvalues, dict):
		J = np.array(list(eigenvalues.keys()), dtype=int)
		energies = np.array(list(eigenvalues.values()), dtype=np.float64)
	else:
		energies = np.asarray(eigenvalues, dtype=np.float64)
		if energies.ndim != 1:
			raise ValueError("Eigenvalues must be 1D.")
		J = np.arange(len(energies))

	if unit not in {"wavenumber", "SI"}:
		raise ValueError("Invalid unit.")

	# ---- Sort by energy ----
	sort_idx = np.argsort(energies)
	energies = energies[sort_idx]
	J = J[sort_idx]

	kB = 0.69503476  # cm^-1/K

	# ---- Energy shifting (numerical stability) ----
	E0 = energies[0]
	Delta = energies - E0

	results = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"T must be > 0. Got {T}")

		beta = 1.0 / (kB * T)

		# ---- Free rotor degeneracy ----
		g = 2 * J + 1

		# ---- Boltzmann weights ----
		weights = g * np.exp(-beta * Delta)

		# ---- Safety cutoff ----
		mask = weights > pop_tol
		weights = weights[mask]
		energies_used = energies[mask]
		J_used = J[mask]

		# ---- Partition function ----
		Z = np.sum(weights)

		# ---- Probabilities ----
		populations = weights / Z

		# ---- Cumulative population ----
		cum_pop = np.cumsum(populations)

		# ---- Adaptive truncation ----
		idx_conv = np.searchsorted(cum_pop, cum_tol)

		populations = populations[:idx_conv + 1]
		cum_pop = cum_pop[:idx_conv + 1]
		energies_used = energies_used[:idx_conv + 1]
		J_used = J_used[:idx_conv + 1]

		# ---- Observables ----
		E_avg = np.dot(populations, energies_used)
		E2_avg = np.dot(populations, energies_used**2)

		Cv_cm1 = kB * beta**2 * (E2_avg - E_avg**2)

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

		results[T] = {
			"temperature_K": T,
			"beta": beta,
			"partition_function": Z,
			"populations": populations,
			"cum_populations": cum_pop,
			"J_levels": J_used,
			"internal_energy": U_out,
			"heat_capacity": Cv_out,
			"levels_used": len(populations),
			"convergence_index": idx_conv,
			"convergence_energy": energies_used[idx_conv],
			"unit": unit,
			"display_unit": display_unit,
			"display_cv_unit": display_cv_unit
		}

	return results

def compute_thermo_vectorized(JM_list, eigenvalues, eigenvectors, temperature_list, unit, pop_tol=1e-16, cum_tol=1-1e-14):
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

	kB = 0.69503476  # cm^-1/K
	E0 = energies[0]
	Delta = energies - E0   # shifted energies (>=0)

	results = {}

	for T in temperature_list:
		if T <= 0:
			raise ValueError(f"Temperature must be > 0 K. Got: {T}")

		beta = 1.0 / (kB * T)

		# ---- Full Boltzmann weights ----
		weights = np.exp(-beta * Delta)

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

		# ---- Orientation ----
		#total = np.sum(weights * pair_sum, axis=1)
		total = np.dot(weights, pair_sum)   # FULL
		cos_theta_avg = (2.0 / Z) * total

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

		results[T] = {
			"temperature_K": T,
			"beta": beta,
			"partition_function": Z,
			"populations_full": populations_full,
			"populations_check": populations_check,
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
					entry = thermo_data[T]
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

def plot_dipole_orientation_comparison(thermo_dict_by_molecule, get_temperature_list, unit_want, out_path):
	"""
	Plots heat capacity vs temperature for multiple molecules together.

	Parameters:
		thermo_dict_by_molecule (dict): { molecule: {(jmax, E): thermo_data} }
		get_temperature_list (function): Function to fetch temperature list for a molecule.
		unit_want (str): Unit for Cv display.
		out_path (str or Path): Path to save combined plot.
	"""
	fig, ax = plt.subplots(figsize=(9, 6))

	styles = {
		"HF":  {"color": "black", "linestyle": "-",  "marker": "o"},
		"HCl": {"color": "black", "linestyle": "--", "marker": "s"},
		"HBr": {"color": "black", "linestyle": "-.", "marker": "D"},
		"HI":  {"color": "black", "linestyle": ":",  "marker": "^"}
	}

	for mol_idx, (molecule, thermo_dict) in enumerate(thermo_dict_by_molecule.items()):
		temperature_list = get_temperature_list(molecule, dipole_orientation=True)

		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		for curve_idx, ((jmax, E), thermo_data) in enumerate(thermo_dict.items()):
			dipole_orientation_values = [thermo_data[T]["dipole_orientation"] for T in temperature_list]

			style = styles[molecule]

			# Smooth line
			plt.plot(
				temperature_list,
				dipole_orientation_values,
				color="black",
				linestyle=style["linestyle"],
				linewidth=2.2,
				label=molecule
			)

			"""
			plt.scatter(
				temperature_list[::5],
				dipole_orientation_values[::5],
				facecolors='none',
				edgecolors='black',
				marker=style["marker"],
				s=28,
				linewidths=1,
			)
			"""

	plt.text(
		0.5, 0.85,
		r"Electric Field:" "\n" fr"$E = {E:.1f}\ \mathrm{{kV/cm}}$",
		transform=plt.gca().transAxes,
		fontsize=14,
		ha='center',
		va='center',
		bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
	)

	# -----------------------------
	# Axis labels
	# -----------------------------
	ax.set_xlabel("Temperature (K)", fontsize=18)
	ax.set_ylabel(r"$\langle \cos\theta \rangle$", fontsize=18)

	# -----------------------------
	# Limits
	# -----------------------------
	ax.set_xlim(-0.5, 100.5)
	#ax.set_ylim(-0.01, 0.8)
	ax.margins(y=0.05)

	# -----------------------------
	# Major ticks
	# -----------------------------
	ax.set_xticks(np.arange(0, 101, 10))
	#ax.yaxis.set_major_locator(MultipleLocator(0.1))
	ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

	# -----------------------------
	# Minor ticks
	# -----------------------------
	ax.xaxis.set_minor_locator(MultipleLocator(2))
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	#ax.yaxis.set_minor_locator(MultipleLocator(0.02))

	# -----------------------------
	# Tick styling (publication quality)
	# -----------------------------
	ax.tick_params(axis='both', which='major',
				   direction='in', length=7, width=1.2,
				   labelsize=18, top=True, right=True)

	ax.tick_params(axis='both', which='minor',
				   direction='in', length=4, width=1.0,
				   top=True, right=True)

	# -----------------------------
	# Spines (frame thickness)
	# -----------------------------
	for spine in ax.spines.values():
		spine.set_linewidth(1.5)

	# -----------------------------
	# Optional: light grid (very subtle)
	# -----------------------------
	#ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.5)
	#ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.4)

	plt.legend(fontsize=18, loc="best")
	# -----------------------------
	# Layout
	# -----------------------------
	plt.tight_layout()

	# Save first, then show
	plt.savefig(out_path, dpi=300)
	print("")
	print(f"[INFO] Combined dipole orientation <cos(theta)>_T plot saved: {out_path}")


def get_ground_state_dipole_orientation(thermo_dict_by_molecule, get_temperature_list):
	"""
	Approximate ground-state dipole orientation using lowest available temperature
	and compare with low-field analytical expression.
	"""

	for molecule, thermo_dict in thermo_dict_by_molecule.items():
		temperature_list = get_temperature_list(molecule, dipole_orientation=True)

		# Flatten if needed
		if len(temperature_list) == 1 and isinstance(temperature_list[0], (list, tuple)):
			temperature_list = temperature_list[0]

		T_min = min(temperature_list)

		print(f"\nMolecule: {molecule} (using T = {T_min} K)")

		# Molecular constants
		B_const = MOLECULE_DATA[molecule]["B_const"]
		dipole_moment = MOLECULE_DATA[molecule]["dipole_moment"]

		for (jmax, E), thermo_data in thermo_dict.items():

			if T_min not in thermo_data:
				raise KeyError(
					f"T={T_min} missing for {molecule}, jmax={jmax}, E={E}"
				)

			# Numerical value
			val_num = thermo_data[T_min]["dipole_orientation"]

			# Convert μE → cm^-1
			potential_strength = convert_dipole_field_energy_to_cm_inv(
				dipole_moment, E
			)

			# Dimensionless parameter
			x = potential_strength / B_const

			# Analytical (low-field expansion)
			val_ana = (x / 3.0) * (1.0 - x**2 / 12.0)

			# Difference
			error = abs(val_num - val_ana)

			Z = thermo_data[T_min]["partition_function"]

			print(
				f"  jmax={jmax:2d}, E={E:10.4f}, x={x:10.4f}, Z={Z:10.4f}  |  "
				f"<cosθ>_num = {val_num: .6f}  |  "
				f"<cosθ>_ana = {val_ana: .6f}  |  "
				f"Δ = {error:.2e}"
			)

