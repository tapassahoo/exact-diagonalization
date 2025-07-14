# io_netcdf.py

import os
import inspect
import sys
import getpass
import socket
import platform
import numpy as np
from datetime import datetime
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
import io
from pkg_utils.utils import whoami
from pkg_utils.config import *
from monomer_linear_rotor.utils import (
	convert_dipole_field_energy_to_cm_inv  # or appropriate import
)

def save_all_quantum_data_to_netcdf(
	file_name: str,
	molecule_name: str,
	max_angular_momentum_quantum_number: int,
	spin_state: str,
	B_const_cm_inv: float,
	potential_strength_cm_inv: float,
	all_quantum_numbers: np.ndarray,
	quantum_numbers_for_spin_state: np.ndarray,
	eigenvalues: np.ndarray,
	eigenvectors: np.ndarray,
	dipole_moment_D: float = None,
	electric_field_kVcm: float = None,
	creator: str = "Dr. Tapas Sahoo",
	software_version: str = "1.0"
) -> None:
	"""
	Save quantum numbers, eigenvalues, and eigenvectors to a NetCDF file.

	Includes metadata, physical constants, and optional dipole-field information.
	"""
	eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
	eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)
	real_eigenvectors = np.real(eigenvectors)
	imag_eigenvectors = np.imag(eigenvectors)

	with Dataset(file_name, "w", format="NETCDF4") as ncfile:
		# Global metadata
		write_metadata(ncfile, spin_state)

		# Scalar attributes with units and long names
		global_attrs = {
			"molecule_name": ("dimensionless", molecule_name, "Name of the linear rigid rotor"),
			"max_angular_momentum_quantum_number": ("dimensionless", max_angular_momentum_quantum_number, "Maximum angular momentum quantum number"),
			"spin_state": ("unitless (string)", spin_state, "Spin isomer type (spinless, ortho, para)"),
			"B_const_cm_inv": ("cm^-1", B_const_cm_inv, "Rotational constant"),
			"potential_strength_cm_inv": ("cm^-1", potential_strength_cm_inv, "Orienting potential strength")
		}

		if dipole_moment_D is not None:
			global_attrs["dipole_moment_D"] = ("Debye", dipole_moment_D, "Dipole moment")
		if electric_field_kVcm is not None:
			global_attrs["electric_field_kVcm"] = ("kV/cm", electric_field_kVcm, "Electric field strength")
		if dipole_moment_D is not None and electric_field_kVcm is not None:
			muE_cm_inv = convert_dipole_field_energy_to_cm_inv(dipole_moment_D, electric_field_kVcm)
			global_attrs["muE_cm_inv"] = ("cm^-1", muE_cm_inv, "Dipole-field interaction energy (mu·E)")

		for key, (unit, value, long_name) in global_attrs.items():
			ncfile.setncattr(key, value)
			ncfile.setncattr(f"{key}_units", unit)
			ncfile.setncattr(f"{key}_long_name", long_name)

		write_quantum_numbers(ncfile, all_quantum_numbers, spin_state, quantum_numbers_for_spin_state)
		write_eigen_data(ncfile, eigenvalues, real_eigenvectors, imag_eigenvectors)

def write_metadata(ncfile: Dataset, spin_state_name: str) -> None:
	"""
	Write metadata and execution environment info into a NetCDF file.
	"""
	username = getpass.getuser()
	hostname = socket.getfqdn()
	try:
		ip_address = socket.gethostbyname(socket.gethostname())
	except socket.gaierror:
		ip_address = "unavailable"

	os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
	python_version = sys.version.replace('\n', ' ')
	architecture = platform.machine()
	cwd = os.getcwd()
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	# --- NetCDF Metadata Block ---
	ncfile.setncattr("title", "Quantum Rotational States and Eigenvalue Data")
	ncfile.setncattr("description",
		f"Eigenvalues and eigenfunctions computed by exact diagonalization of the analytical "
		f"Hamiltonian for a polar, rigid, linear rotor in an external electric field. "
		f"The dipole–field interaction is treated explicitly, and the potential energy matrix "
		f"elements are evaluated using Wigner's 3-j symbols. "
		f"Spin isomer: '{spin_state_name}'."
	)
	ncfile.setncattr("source", "pkg_monomer_rotor/main.py")
	ncfile.setncattr("institution", "National Institute of Technology Raipur")
	ncfile.setncattr("history", f"Created on {timestamp} by {username} on host '{hostname}'")
	ncfile.setncattr("license", "Data intended for academic and research use only.")
	ncfile.setncattr("conventions", "CF-1.6")

	# --- Execution Environment Metadata ---
	ncfile.setncattr("creator_name", "Dr. Tapas Sahoo")
	ncfile.setncattr("creator_host", hostname)
	ncfile.setncattr("creator_ip", ip_address)
	ncfile.setncattr("operating_system", os_info)
	ncfile.setncattr("architecture", architecture)
	ncfile.setncattr("python_version", python_version)
	ncfile.setncattr("working_directory", cwd)
	ncfile.setncattr("creation_time", timestamp)


def write_quantum_numbers(ncfile, all_qn, spin_state, filtered_qn):
	n_all, n_comp = all_qn.shape
	n_filt = filtered_qn.shape[0]

	ncfile.createDimension("n_qn_all", n_all)
	ncfile.createDimension("n_qn_allowed", n_filt)
	ncfile.createDimension("qn_components", n_comp)

	var_all = ncfile.createVariable("all_quantum_numbers", "i4", ("n_qn_all", "qn_components"))
	var_all[:, :] = all_qn
	var_all.units = "dimensionless"
	var_all.long_name = "Complete list of quantum numbers (e.g., J, M or l, m)"

	var_filt = ncfile.createVariable("quantum_numbers_for_spin_state", "i4", ("n_qn_allowed", "qn_components"))
	var_filt[:, :] = filtered_qn
	var_filt.units = "dimensionless"
	var_filt.long_name = f"Quantum numbers allowed for spin state: {spin_state}"


def write_eigen_data(ncfile, eigenvalues, real_evecs, imag_evecs):
	"""
	Save eigenvalues and eigenvectors to a NetCDF file with shape validation.

	Parameters:
		ncfile (netCDF4.Dataset): Open NetCDF file.
		eigenvalues (np.ndarray): Array of eigenvalues (shape: n_states,)
		real_evecs (np.ndarray): Real part of eigenvectors (shape: n_states, n_basis)
		imag_evecs (np.ndarray): Imaginary part of eigenvectors (shape: n_states, n_basis)
	"""

	# Convert inputs to arrays and validate types
	eigenvalues = np.asarray(eigenvalues)
	real_evecs = np.asarray(real_evecs)
	imag_evecs = np.asarray(imag_evecs)

	# --- Shape Validation ---
	if real_evecs.shape != imag_evecs.shape:
		raise ValueError(f"[ERROR] Shape mismatch: real_evecs shape {real_evecs.shape} ≠ imag_evecs shape {imag_evecs.shape}")

	n_basis, n_states = real_evecs.shape

	if eigenvalues.shape != (n_states,):
		raise ValueError(
			f"[ERROR] Shape mismatch: eigenvalues should have shape ({n_states},) but got {eigenvalues.shape}"
		)

	# --- Create Dimensions ---
	ncfile.createDimension("n_states", n_states)
	ncfile.createDimension("n_basis", n_basis)

	# --- Store Eigenvalues ---
	var_eval = ncfile.createVariable("eigenvalues", "f8", ("n_states",))
	var_eval[:] = eigenvalues
	var_eval.units = "cm^-1"
	var_eval.long_name = "Eigenvalues (energy levels)"

	# --- Store Eigenvectors ---
	var_real = ncfile.createVariable("real_eigenvectors", "f8", ("n_basis", "n_states"))
	var_real[:, :] = real_evecs
	var_real.units = "dimensionless"
	var_real.long_name = "Real part of eigenvectors"

	var_imag = ncfile.createVariable("imag_eigenvectors", "f8", ("n_basis", "n_states"))
	var_imag[:, :] = imag_evecs
	var_imag.units = "dimensionless"
	var_imag.long_name = "Imaginary part of eigenvectors"

def read_all_attributes(filename: str, show_variables: bool = True) -> None:
	"""
	Print global metadata, additional attributes, dimensions, and variable-level attributes from a NetCDF file.

	Parameters:
		filename (str): Path to the NetCDF file.
		show_variables (bool): Whether to display variable-level attributes and shapes.
	"""
	important_keys = [
		"title", "description", "source", "institution", "history", "license", "conventions", 
		"creator_name", "creator_host", "creator_ip", "operating_system", "architecture", "python_version", 
		"working_directory", "creation_time"
	]


	try:
		with Dataset(filename, 'r') as ncfile:
			print("\n--- Global Metadata ---")
			print("-" * 60)
			for key in important_keys:
				if key in ncfile.ncattrs():
					print(f"{key:<30}: {ncfile.getncattr(key)}")

			print("\n--- Additional Global Attributes ---")
			print("-" * 60)
			for attr in ncfile.ncattrs():
				if attr not in important_keys:
					print(f"{attr:<30}: {ncfile.getncattr(attr)}")

			print("\n--- Dimensions ---")
			print("-" * 60)
			for dim_name, dim in ncfile.dimensions.items():
				print(f"{dim_name:<30}: size = {len(dim)}")

			if show_variables:
				print("\n--- Variable-wise Attributes and Shapes ---")
				print("-" * 60)
				for var_name, var in ncfile.variables.items():
					shape_str = ", ".join(f"{s}" for s in var.shape)
					print(f"\nVariable: {var_name}")
					print(f"  shape: ({shape_str})")
					print(f"  dtype: {var.dtype}")
					for attr in var.ncattrs():
						print(f"  {attr:<28}: {var.getncattr(attr)}")

			print(f"\n[Status] Attribute inspection of '{filename}' completed successfully.\n")

	except FileNotFoundError:
		print(f"[ERROR] File not found - {filename}")
	except Exception as e:
		print(f"[ERROR] Could not read NetCDF attributes. Details: {e}")


def inspect_variable(filename, variable_name, show_data=True, show_plot=False, slice_index=0, max_elements=100):
	"""
	Inspect a specific variable in a NetCDF file with support for 1D, 2D, and 3D data.

	Parameters:
		filename (str): Path to the NetCDF file.
		variable_name (str): Name of the variable to inspect.
		show_data (bool): Whether to print the data values.
		show_plot (bool): Whether to display a heatmap (2D slice only).
		slice_index (int): Index along the first axis for 3D tensors.
		max_elements (int): Maximum number of elements to display for large 1D data.
	"""
	try:
		with Dataset(filename, 'r') as nc:
			if variable_name not in nc.variables:
				print(f"[X] Variable '{variable_name}' not found in the file.")
				return

			var = nc.variables[variable_name]
			data = var[:]
			dims = var.dimensions

			# Metadata summary
			print(f"\n[INFO] Variable '{variable_name}'")
			print("=" * 70)
			print(f"[ ] Dimensions    {dims}")
			print(f"[ ] Shape         {var.shape}")
			print(f"[ ] Data type     {var.dtype}")
			units = getattr(var, "units", "[not defined]")
			print(f"[ ] Units         {units}")

			for attr in var.ncattrs():
				if attr != "units":
					print(f"[ ] {attr:<13} {getattr(var, attr)}")

			# Display data
			if show_data:
				print("\n[Data]")
				print("-" * 70)
				if data.ndim == 1:
					if data.size > max_elements:
						print(f"  [Showing first {max_elements} of {data.size} elements]")
						print(data[:max_elements])
					else:
						print("  ", data)
				elif data.ndim == 2:
					with np.printoptions(precision=4, suppress=True, linewidth=150):
						for row in data:
							print("  " + "  ".join(f"{val:10.4f}" for val in row))
				elif data.ndim == 3:
					print(f"  [Showing 2D slice at index {slice_index} along axis 0]\n")
					if slice_index >= data.shape[0]:
						print(f"[ERROR] Invalid slice_index {slice_index}, max allowed: {data.shape[0] - 1}")
						return
					slice_2d = data[slice_index]
					with np.printoptions(precision=4, suppress=True, linewidth=150):
						for row in slice_2d:
							print("  " + "  ".join(f"{val:10.4f}" for val in row))
				else:
					print("[ERROR] Data with ndim > 3 is not supported for printing")

			# Optional heatmap
			if show_plot:
				if data.ndim == 2:
					plot_2d_heatmap(data, variable_name, dims, units)
				elif data.ndim == 3:
					if slice_index < data.shape[0]:
						plot_2d_heatmap(data[slice_index], f"{variable_name} (slice {slice_index})", dims[1:], units)
					else:
						print(f"[ERROR] Invalid slice_index {slice_index} for shape {data.shape}")

			print("=" * 70 + "\n")

	except FileNotFoundError:
		print(f"[ERROR] File not found: {filename}")
	except Exception as e:
		print(f"[ERROR] Error while inspecting variable: {e}")

def plot_2d_heatmap(matrix, title, dims, units):
	"""
	Helper function to plot a 2D heatmap of a matrix.
	"""
	plt.figure(figsize=(8, 6))
	im = plt.imshow(matrix, cmap='viridis', aspect='auto')
	plt.colorbar(im, label=units)
	plt.title(f"Heatmap of '{title}'")
	plt.xlabel(f"{dims[1]}")
	plt.ylabel(f"{dims[0]}")
	plt.tight_layout()
	plt.show()

