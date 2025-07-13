import argparse
import sys
import os
from pathlib import Path
import subprocess
from typing import List
import csv
import logging
from itertools import product
from datetime import datetime
import shutil
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


MOLECULE_DATA = {
	"HF": {"dipole_moment": 1.83},
	"HCl": {"dipole_moment": 1.03},
	"HBr": {"dipole_moment": 0.78},
}
allowed_spin_types = ["spinless", "ortho", "para"]

def parse_arguments():
	parser = argparse.ArgumentParser(
		description=(
			"Submit rotor simulations for various angular momentum cutoffs and field strengths.\n\n"
			"Examples:\n"
			"  python submit_rotor_jobs_all_at_once.py ortho --molecule HF\n"
			"  python submit_rotor_jobs_all_at_once.py para --dipole-moment 1.75 --electric-fields 50 100 --jmax-values 6 8 10\n"
			"  python submit_rotor_jobs_all_at_once.py spinless --molecule HCl --use-potential --electric-fields 10 20 30\n"
		),
		epilog="Developed by Dr. Tapas Sahoo — Quantum Molecular Dynamics Group",
		formatter_class=argparse.RawTextHelpFormatter
	)

	# --- Positional argument ---
	parser.add_argument(
		"spin_type",
		type=str,
		choices=allowed_spin_types,
		help="Spin isomer type: 'spinless', 'ortho', or 'para'."
	)

	# --- Optional arguments ---
	parser.add_argument(
		"--molecule", type=str,
		help="Molecule name (e.g., HF, HCl) to auto-fill dipole moment (μ in Debye)."
	)

	parser.add_argument(
		"--dipole-moment", type=float, default=None,
		help="Dipole moment μ in Debye (overrides value from --molecule if provided)."
	)

	parser.add_argument(
		"--electric-fields", type=float, nargs='+', default=[100.0],
		help="List of electric field strengths (in kV/cm if using μ·E, or cm⁻¹ if using --use-potential)."
	)

	parser.add_argument(
		"--jmax-values", type=int, nargs='+', default=[6, 8, 10, 12],
		help="List of ℓ_max values (maximum angular momentum quantum number). Default: 6 8 10 12"
	)

	parser.add_argument(
		"--use-potential", action="store_true",
		help="Interpret electric-fields as V(θ) (in cm⁻¹) instead of computing μ·E."
	)

	parser.add_argument(
		"--dry-run", action="store_true",
		help="Print the commands without executing them."
	)

	args = parser.parse_args()

	# --- Auto-resolve dipole moment if needed ---
	if not args.use_potential:
		if args.molecule:
			mol = args.molecule.strip().upper()
			if mol not in MOLECULE_DATA:
				print(f"[ERROR] Unknown molecule '{mol}'. Choose from: {', '.join(MOLECULE_DATA.keys())}")
				sys.exit(1)
			if args.dipole_moment is None:
				args.dipole_moment = MOLECULE_DATA[mol]["dipole_moment"]

		if args.dipole_moment is None:
			print("[ERROR] Must provide --dipole-moment or --molecule (unless using --use-potential).")
			sys.exit(1)

	return args


def setup_logging(log_file):
	os.makedirs(os.path.dirname(log_file), exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		handlers=[
			logging.FileHandler(log_file, mode='w'),
			logging.StreamHandler()
		]
	)

def get_parameter_combinations(jmax_values, dipole_moment, electric_field_values, potential_strength_values):
	use_dipole_field = dipole_moment is not None and electric_field_values
	if use_dipole_field:
		return list(product(jmax_values, electric_field_values)), True
	else:
		return list(product(jmax_values, potential_strength_values)), False

def preview_job_settings(molecule, spin_type, dipole, param_combinations, output_dir, csv_path, script_name):
	logging.info("==========================================")
	logging.info(f"Launching {molecule} rotor job submissions")
	logging.info("==========================================")
	logging.info(f"Molecule              : {molecule}")
	logging.info(f"Spin type             : {spin_type}")
	logging.info(f"Dipole moment (D)     : {dipole}")
	logging.info(f"Script to execute     : {os.path.join(output_dir, script_name)}")
	logging.info(f"Total job combinations: {len(param_combinations)}")
	logging.info(f"Base output directory : {output_dir}")
	logging.info(f"Summary CSV will be at: {csv_path}")
	logging.info("------------------------------------------")

def build_command(jmax, value, use_dipole, dipole, output_dir, spin_type):
	args = [
		"python", script_name,
		str(jmax), spin_type,
		"--output-dir", output_dir
	]
	if use_dipole:
		args += ["--dipole-moment", str(dipole), "--electric-field", str(value)]
	else:
		args += ["--potential-strength", str(value)]
	return args

def check_job_status(tag, job_dir, jmax, spin_type, info_str, summary_rows):
	stdout_path = os.path.join(job_dir, f"{tag}.stdout")
	status_file = os.path.join(job_dir, "status.txt")

	job_completed = False
	if os.path.exists(stdout_path):
		with open(stdout_path, "r") as f:
			contents = f.read()
			if "HURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES" in contents:
				job_completed = True

	current_status = None
	if os.path.exists(status_file):
		with open(status_file) as f:
			current_status = f.read().strip().upper()

	if job_completed or current_status == "COMPLETED":
		print(f"[Skipping ] {tag}: status = COMPLETED")
		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": spin_type,
			"Field/Interaction": info_str,
			"Status": "COMPLETED",
			"PID": "-"
		})
		return True

	if current_status == "RUNNING":
		print(f"[Skipping ] {tag}: status = RUNNING")
		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": spin_type,
			"Field/Interaction": info_str,
			"Status": "RUNNING",
			"PID": "-"
		})
		return True

	return False

def write_summary_csv(rows, csv_path):
	fieldnames = ["Job Name", "Max Angular Momentum", "Spin Type", "Field/Interaction", "Status", "PID"]
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

if False:
	def build_and_run_command(
		jmax: int,
		electric_field: float,
		use_dipole: bool,
		dipole_moment: float,
		output_dir: str,
		spin_type: str,
		script_name: str = "main.py",
		dry_run: bool = False,
		execute: bool = False
	) -> List[str]:
		"""
		Build and optionally execute a rotor diagonalization command.

		Parameters:
			jmax (int): Maximum angular momentum quantum number (ℓ_max).
			electric_field (float): Electric field strength in kV/cm (if use_dipole=True)
									or potential strength V(θ) in cm⁻¹ (if use_dipole=False).
			use_dipole (bool): If True, uses dipole moment and electric field to compute potential.
							   If False, uses --potential-strength directly.
			dipole_moment (float): Dipole moment μ in Debye.
			output_dir (str): Directory to store simulation output.
			spin_type (str): Spin isomer type: 'spinless', 'ortho', or 'para'.
			script_name (str): Name of the rotor simulation script.
			dry_run (bool): If True, only print the command (do not execute).
			execute (bool): If True, execute the command using subprocess.

		Returns:
			List[str]: The constructed command-line argument list.
		"""
		# Ensure the output directory exists
		Path(output_dir).mkdir(parents=True, exist_ok=True)

		# Base command structure
		cmd = [
			"python3", script_name,
			str(jmax),
			spin_type,
			"--output-dir", output_dir
		]

		# Include dipole + field or direct potential strength
		if use_dipole:
			cmd += [
				"--dipole-moment", str(dipole_moment),
				"--electric-field", str(electric_field)
			]
		else:
			cmd += [
				"--potential-strength", str(electric_field)
			]

		# Print command in dry-run mode
		if dry_run:
			print(f"[DRY RUN] Command preview:")
			print("  " + " ".join(cmd))
			print("-" * 80)

		# Execute the command if requested
		if execute:
			try:
				subprocess.run(cmd, check=True)
			except subprocess.CalledProcessError as e:
				print(f"[ERROR] Command failed with exit code {e.returncode}: {e.cmd}")

		return cmd

def build_job_command(jmax, spin_type, molecule, efield):
	"""
	Construct the command to run the rotor diagonalization script.
	"""
	cmd = [
		"python3", "main.py",
		str(jmax),
		spin_type,
		"--molecule", molecule,
		"--electric-field", efield
	]
	return cmd


def main():

	args = parse_arguments()

	# --- Sweep Configuration ---
	script_name = "main.py"
	potential_strength_values = [0.1, 0.5]
	jmax_values =list(range(10, 11, 5)) 
	electric_field_values = [0.1] + list(range(20, 21, 20))# [10, 50, 100, 150, 200]   # in kV/cm
	use_dipole = True								 # False if using --potential-strength instead
	molecule_tag = args.molecule if args.molecule else "custom"

	# --- Execution Banner ---
	mode = "DRY RUN" if args.dry_run else "EXECUTION"
	print(f"[{mode}] Submitting jobs for: molecule={molecule_tag}, spin={args.spin_type}")
	print("=" * 80)

	# --- Construct output directory structure ---
	os.makedirs("output", exist_ok=True)
	# Copy script to job_dir
	script_basename = os.path.basename(script_name)

	for jmax in jmax_values:
		for efield in electric_field_values:

			# Use the molecule and spin arguments as-is, preserving original casing
			subdir_name = f"{args.spin_type}_{args.molecule}_jmax{jmax}_efield{efield}kV_per_cm"

			# Final output path
			output_root_dir = os.path.join("output", subdir_name)
			os.makedirs(output_root_dir, exist_ok=True)
			script_dest_path = os.path.join(output_root_dir, script_basename)
			shutil.copy2(script_name, script_dest_path)
			script_path_in_job_dir = os.path.join(output_root_dir, script_basename)


			"""
			build_and_run_command(
				jmax=jmax,
				electric_field=efield,
				use_dipole=use_dipole,
				dipole_moment=args.dipole_moment,
				output_dir=output_root_dir,
				spin_type=args.spin_type,
				script_name = "main.py",
				dry_run=args.dry_run,
				execute=not args.dry_run
			)
			"""

			cmd=build_job_command(jmax, args.spin_type, args.molecule, efield)
			print(cmd)

	print("=" * 80)
	print("Job submission completed.")

	csv_path = os.path.join(output_root_dir, "job_summary.csv")
	log_file = os.path.join(output_root_dir, "job_submission.log")

	setup_logging(log_file)

	param_combinations, use_dipole = get_parameter_combinations(jmax_values, args.dipole_moment, electric_field_values, potential_strength_values)
	preview_job_settings(args.molecule, args.spin_type, args.dipole_moment, param_combinations, output_dir, csv_path, script_name)
	whoami()

	summary_rows = []

	for jmax, secondary_param in param_combinations:
		if use_dipole:
			E = secondary_param
			tag = f"{args.spin_type}_{args.molecule}_lmax_{jmax}_dipole_{args.dipole:.2f}D_E_{E:.2f}kVcm".replace(".", "_")
			cmd = [
				"python3", script_path_in_job_dir,
				str(jmax),
				args.spin_type,
				"--dipole-moment", str(args.dipole),
				"--electric-field", str(E),
				"--output-dir", os.path.join(output_dir, tag)
			]
			info_str = f"Dipole moment = {args.dipole} D, E = {E} kV/cm"
		else:
			V = secondary_param
			tag = f"{args.molecule}_Vfield_{V:.2f}cm1_Jmax_{jmax}_{args.spin_type}".replace(".", "_")
			cmd = [
				"python3", script_path_in_job_dir,
				str(V),
				str(jmax),
				args.spin_type,
				"--output-dir", os.path.join(output_dir, tag)
			]
			info_str = f"V = {V} cm⁻¹"

		job_dir = os.path.join(output_dir, tag)
		os.makedirs(job_dir, exist_ok=True)

		status = determine_job_status(stdout_path, status_file)
		if status in ["COMPLETED", "RUNNING"]:
			print(f"[Skipping ] {tag}: status = {status}")
			summary_rows.append({
				"Job Name": tag,
				"Max Angular Momentum": jmax,
				"Spin Type": args.spin_type,
				"Field/Interaction": info_str,
				"Status": status,
				"PID": "-"
			})
			continue
			continue

		stdout_path = os.path.join(job_dir, f"{tag}.stdout")
		stderr_path = os.path.join(job_dir, f"{tag}.stderr")

		cmd_str = " ".join(cmd)

		if args.dry_run:
			print(f"\n\n[DRY RUN ] Job Name	   : {tag}")
			print(f"			Command	   : {cmd_str}")
			print(f"			Stdout Path   : {stdout_path}")
			print(f"			Stderr Path   : {stderr_path}")

			summary_rows.append({
				"Job Name": tag,
				"Max Angular Momentum": jmax,
				"Spin Type": args.spin_type,
				"Field/Interaction": info_str,
				"Status": "DRY-RUN",
				"PID": "-"
			})
			continue


		print(f"\n\n[Launching] Job: {tag}")
		print(f"  > Command	  : {cmd_str}")
		print(f"  > Stdout path  : {stdout_path}")
		print(f"  > Stderr path  : {stderr_path}")

		run_script_path = os.path.join(job_dir, f"{tag}_run_command.sh")
		with open(run_script_path, "w") as f:
			f.write("#!/bin/bash\n")
			f.write(cmd_str + "\n")

		with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
			subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)

		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": args.spin_type,
			"Field/Interaction": info_str,
			"Status": "SUBMITTED",
			"PID": "-"
		})

	write_summary_csv(summary_rows, csv_path)
	logging.info("All jobs processed. Summary written to CSV.")

	print("==========================================")
	print("HURRAY ALL JOBS SUBMITTED SUCCESSFULLY")
	print(f"Output directory: {output_dir}")
	print(f"Summary CSV: {csv_path}")
	print("==========================================")

if __name__ == "__main__":
	main()
