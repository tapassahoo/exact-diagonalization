import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from netCDF4 import Dataset

from pkg_utils.utils import whoami
from pkg_utils.env_report import whom

def read_eigenvalues_from_netcdf(filename):
	with Dataset(filename, 'r') as nc:
		return np.sort(nc.variables["eigenvalues"][:])

def compute_relative_differences(eigenvalue_lists, level_wanted=None):
	sorted_js = sorted(eigenvalue_lists.keys())
	rel_diffs = {}
	min_len = min(len(arr) for arr in eigenvalue_lists.values())

	levels = [level_wanted] if level_wanted is not None else range(min_len)

	for level in levels:
		rel_diffs[level] = {}
		for i in range(1, len(sorted_js)):
			j1, j2 = sorted_js[i - 1], sorted_js[i]
			e1, e2 = eigenvalue_lists[j1][level], eigenvalue_lists[j2][level]
			rel_diffs[level][j2] = abs((e2 - e1) / e2) if e2 != 0 else np.nan
	return rel_diffs

if False:
	def plot_eigenvalue_convergence(jmax_list, file_template, num_levels_to_show=5, threshold=1e-4, level_wanted=None):
		"""
		Plot convergence of the lowest few eigenvalues vs J_max, and print relative errors.

		Parameters:
			jmax_list (list of int): List of J_max values.
			file_template (str): Template path for NetCDF files with {jmax}.
			num_levels_to_show (int): Number of eigenvalues to track (if level_wanted is None).
			threshold (float): Relative error threshold for convergence warning.
			level_wanted (int or None): If specified, only this eigenvalue level is analyzed and plotted.
		"""
		eigenvalues_by_jmax = {}

		for jmax in jmax_list:
			filename = file_template.format(jmax=jmax)
			if os.path.isfile(filename):
				eigenvalues_by_jmax[jmax] = read_eigenvalues_from_netcdf(filename)
			else:
				print(f"[Warning] File not found: {filename}")

		if not eigenvalues_by_jmax:
			print("[Error] No eigenvalue files were successfully loaded.")
			return

		min_levels = min(len(arr) for arr in eigenvalues_by_jmax.values())
		if level_wanted is not None and (level_wanted < 0 or level_wanted >= min_levels):
			print(f"[ERROR] level_wanted = {level_wanted} is out of bounds. Available levels: 0 to {min_levels - 1}.")
			return

		rel_diffs = compute_relative_differences(eigenvalues_by_jmax, level_wanted=level_wanted)

		print("\nConvergence Report (relative differences):")
		levels_to_show = [level_wanted] if level_wanted is not None else range(min(num_levels_to_show, min_levels))

		for level in levels_to_show:
			print(f"\n  Level {level}:")
			for jmax in sorted(jmax_list)[1:]:
				if jmax in rel_diffs[level]:
					rd = rel_diffs[level][jmax]
					status = "OK" if rd < threshold else "Not converged"
					print(f"	Δ(J={jmax}) = {rd:.2e} [{status}]")

		# --- Plotting ---
		plt.figure(figsize=(10, 6))

		for level in levels_to_show:
			js, energies = [], []
			for j in jmax_list:
				if j in eigenvalues_by_jmax and level < len(eigenvalues_by_jmax[j]):
					js.append(j)
					energies.append(eigenvalues_by_jmax[j][level])

			plt.plot(js, energies, marker='o', label=fr"$E_{{{level}}}$")

			# Annotate each point
			for j_val, e_val in zip(js, energies):
				plt.annotate(f"{e_val:.6f}", xy=(j_val, e_val), xytext=(0, 5),
							 textcoords='offset points', fontsize=8,
							 ha='center', color='darkblue')

		plt.xlabel(r"$J_{\mathrm{max}}$", fontsize=12)
		plt.ylabel("Eigenvalue (cm$^{-1}$)", fontsize=12)
		title_label = f"level {level_wanted}" if level_wanted is not None else f"lowest {num_levels_to_show} levels"
		plt.title(f"Convergence of {title_label} with $J_{{\\mathrm{{max}}}}$", fontsize=14, weight='bold')

		ax = plt.gca()
		ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

		plt.grid(True, linestyle='--', alpha=0.4)
		plt.legend()
		plt.tight_layout()
		plt.show()


def plot_eigenvalue_convergence(jmax_list, file_template, num_levels_to_show=5, threshold=1e-4, level_wanted=None):
	"""
	Print convergence of eigenvalues vs J_max, and plot:
	- All levels if num_levels_to_show <= 5
	- Only the specified level if level_wanted is given

	Parameters:
		jmax_list (list of int): List of J_max values.
		file_template (str): Template path for NetCDF files with {jmax}.
		num_levels_to_show (int): Number of eigenvalues to track if level_wanted is None.
		threshold (float): Relative error threshold.
		level_wanted (int or None): If set, only this level is analyzed and plotted.
	"""
	import os
	import matplotlib.pyplot as plt

	eigenvalues_by_jmax = {}

	for jmax in jmax_list:
		filename = file_template.format(jmax=jmax)
		if os.path.isfile(filename):
			eigenvalues = read_eigenvalues_from_netcdf(filename)
			eigenvalues_by_jmax[jmax] = eigenvalues
		else:
			print(f"[Warning] File not found: {filename}")

	if not eigenvalues_by_jmax:
		print("[ERROR] No valid NetCDF files found.")
		return

	# Determine how many levels are available
	min_levels = min(len(vals) for vals in eigenvalues_by_jmax.values())

	# Validate requested level
	if level_wanted is not None:
		if level_wanted < 0 or level_wanted >= min_levels:
			print(f"[ERROR] level_wanted = {level_wanted} is out of bounds (0 to {min_levels - 1}).")
			return
		levels_to_show = [level_wanted]
	else:
		levels_to_show = range(min(num_levels_to_show, min_levels))

	# Compute relative differences
	rel_diffs = compute_relative_differences(eigenvalues_by_jmax, level_wanted=level_wanted)

	# --- Print Convergence Report ---
	print("\nConvergence Report (relative differences):")
	for level in levels_to_show:
		print(f"\n  Level {level}:")
		for jmax in sorted(jmax_list)[1:]:
			if jmax in rel_diffs[level]:
				rd = rel_diffs[level][jmax]
				status = "OK" if rd < threshold else "Not converged"
				print(f"	Δ(J={jmax}) = {rd:.2e} [{status}]")

	# --- Skip Plotting if too many levels requested ---
	if level_wanted is None and num_levels_to_show > 5:
		print("\n[INFO] Plotting skipped: too many levels for clear visualization.")
		return

	# --- Plotting ---
	plt.figure(figsize=(10, 6))

	for level in levels_to_show:
		js, energies = [], []
		for j in jmax_list:
			if j in eigenvalues_by_jmax and level < len(eigenvalues_by_jmax[j]):
				js.append(j)
				energies.append(eigenvalues_by_jmax[j][level])
		plt.plot(js, energies, marker='o', label=rf"$E_{{{level}}}$")
		for x, y in zip(js, energies):
			plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

	plt.xlabel(r"$J_{\mathrm{max}}$", fontsize=12)
	plt.ylabel(r"Eigenvalue (cm$^{-1}$)", fontsize=12)
	if level_wanted is not None:
		title_label = rf"$E_{{{level_wanted}}}$"
	else:
		title_label = rf"Lowest {num_levels_to_show} Levels"
	plt.title(f"Convergence of {title_label} with $J_{{\\mathrm{{max}}}}$", fontsize=14, weight='bold')
	plt.grid(True, linestyle='--', alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.show()
