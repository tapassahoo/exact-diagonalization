
import os
import subprocess
import csv
import logging
import argparse
from itertools import product
from datetime import datetime

electric_field_kVcm_list = [0.1] + list(range(20, 201, 20))
potential_strength_list = [0.1, 0.5]
max_angular_momentum_list = list(range(10, 31, 5))
script_name = "monomer_rotor_real_basis_diagonalization.py"
allowed_spin_types = {"spinless", "ortho", "para"}

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

def get_parameter_combinations(dipole_moment):
	use_dipole_field = dipole_moment is not None and electric_field_kVcm_list
	if use_dipole_field:
		return list(product(max_angular_momentum_list, electric_field_kVcm_list)), True
	else:
		return list(product(max_angular_momentum_list, potential_strength_list)), False

def preview_job_settings(molecule, spin_type, dipole, param_combinations, output_dir, csv_path):
	logging.info("==========================================")
	logging.info(f"Launching {molecule} rotor job submissions")
	logging.info("==========================================")
	logging.info(f"Molecule              : {molecule}")
	logging.info(f"Spin type             : {spin_type}")
	logging.info(f"Dipole moment (D)     : {dipole}")
	logging.info(f"Script to execute     : {script_name}")
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

def main():
	parser = argparse.ArgumentParser(description="Submit all rotor jobs at once in background")
	parser.add_argument("--molecule", required=True, help="Name of the molecule (e.g., HF, HCl)")
	parser.add_argument("--dipole", type=float, help="Dipole moment in Debye (if any)")
	parser.add_argument("--spin-type", default="spinless", choices=allowed_spin_types, help="Spin type (default: spinless)")
	parser.add_argument("--dry-run", action="store_true", help="Preview commands without executing jobs")
	args = parser.parse_args()

	if args.dipole is not None and args.dipole < 0:
		raise ValueError("Dipole moment must be non-negative.")

	output_dir = f"output_{args.spin_type}_{args.molecule}_monomer_in_field"
	csv_path = os.path.join(output_dir, "job_summary.csv")
	log_file = os.path.join(output_dir, "job_submission.log")

	setup_logging(log_file)

	param_combinations, use_dipole = get_parameter_combinations(args.dipole)
	preview_job_settings(args.molecule, args.spin_type, args.dipole, param_combinations, output_dir, csv_path)

	summary_rows = []

	for jmax, value in param_combinations:
		tag = f"{args.molecule}_{args.spin_type}_lmax_{jmax}_{'E' if use_dipole else 'V'}_{value:.2f}"
		info_str = f"{'Field' if use_dipole else 'Potential'} = {value:.2f}"
		job_dir = output_dir

		if check_job_status(tag, job_dir, jmax, args.spin_type, info_str, summary_rows):
			continue

		stdout_path = os.path.join(job_dir, f"{tag}.stdout")
		stderr_path = os.path.join(job_dir, f"{tag}.stderr")

		cmd = build_command(jmax, value, use_dipole, args.dipole, output_dir, args.spin_type)
		cmd_str = " ".join(cmd)

		if args.dry_run:
			print(f"[Dry Run  ] Would launch: {tag}")
			print(f"  > Command   : {cmd_str}")
			print(f"  > Stdout path  : {stdout_path}")
			print(f"  > Stderr path  : {stderr_path}")
			summary_rows.append({
				"Job Name": tag,
				"Max Angular Momentum": jmax,
				"Spin Type": args.spin_type,
				"Field/Interaction": info_str,
				"Status": "DRY-RUN",
				"PID": "-"
			})
			continue

		print(f"[Launching] Job: {tag}")
		print(f"  > Command   : {cmd_str}")
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

if __name__ == "__main__":
	main()
