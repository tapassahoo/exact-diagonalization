import argparse
import sys
import os
import subprocess
import csv
import logging
from itertools import product
from datetime import datetime
import shutil
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


electric_field_kVcm_list = [0.1] + list(range(20, 21, 20))
potential_strength_list = [0.1, 0.5]
max_angular_momentum_list = list(range(10, 11, 5))
script_name = "monomer_rotor_real_basis_diagonalization.py"

# Allowed spin types
allowed_spin_types = ["spinless", "ortho", "para"]

# Molecular constants database (extend as needed)
MOLECULE_DATA = {
	"HF":  {"dipole_moment": 1.83, "B_const": 20.559},
	"HCl": {"dipole_moment": 1.03, "B_const": 10.44},
	"HBr": {"dipole_moment": 0.78, "B_const": 8.467},
	"HI":  {"dipole_moment": 0.38, "B_const": 6.51},
	"CO":  {"dipole_moment": 0.112, "B_const": 1.9225},
}

import argparse
import sys

# Allowed spin types
allowed_spin_types = ["spinless", "ortho", "para"]

# Molecular database (dipole moments only)
MOLECULE_DATA = {
	"HF":  {"dipole_moment": 1.83},
	"HCl": {"dipole_moment": 1.03},
	"HBr": {"dipole_moment": 0.78},
	"HI":  {"dipole_moment": 0.38},
	"CO":  {"dipole_moment": 0.112},
}

def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Submit all rotor jobs at once in background.",
		epilog="Example: python submit_rotor_jobs_all_at_once.py --molecule HF --spin-type ortho"
	)

	parser.add_argument("--molecule", type=str,
						help="Molecule name (e.g., 'HF', 'HCl') to auto-fill dipole moment (μ in Debye).")

	parser.add_argument("--dipole-moment", type=float, default=None,
						help="Dipole moment μ (in Debye). Overrides value from --molecule if given.")

	parser.add_argument("--spin-type", default="spinless", choices=allowed_spin_types,
						help="Spin isomer type: spinless, ortho, or para (default: spinless).")

	parser.add_argument("--dry-run", action="store_true",
						help="Print job commands without executing them.")

	args = parser.parse_args()

	# --- Molecule-based dipole moment setup ---
	if args.molecule:
		mol = args.molecule.strip().upper()
		if mol not in MOLECULE_DATA:
			print(f"[ERROR] Molecule '{mol}' not found in database. Available: {', '.join(MOLECULE_DATA.keys())}")
			sys.exit(1)

		# Set dipole moment only if not manually provided
		if args.dipole_moment is None:
			args.dipole_moment = MOLECULE_DATA[mol]["dipole_moment"]

	if args.dipole_moment is None:
		print("[ERROR] Provide either --molecule or --dipole-moment.")
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
	logging.info(f"Molecule			  : {molecule}")
	logging.info(f"Spin type			 : {spin_type}")
	logging.info(f"Dipole moment (D)	 : {dipole}")
	logging.info(f"Script to execute	 : {os.path.join(output_dir, script_name)}")
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

	args = parse_arguments()
	print("Dipole Moment:", args.dipole_moment)
	print("Spin Type:", args.spin_type)
	whoami()

	output_dir = f"output_{args.spin_type}_{args.molecule}_monomer_in_field"
	os.makedirs(output_dir, exist_ok=True)
	# Copy script to job_dir
	script_basename = os.path.basename(script_name)
	script_dest_path = os.path.join(output_dir, script_basename)
	shutil.copy2(script_name, script_dest_path)
	script_path_in_job_dir = os.path.join(output_dir, script_basename)

	csv_path = os.path.join(output_dir, "job_summary.csv")
	log_file = os.path.join(output_dir, "job_submission.log")

	setup_logging(log_file)

	param_combinations, use_dipole = get_parameter_combinations(args.dipole)
	preview_job_settings(args.molecule, args.spin_type, args.dipole, param_combinations, output_dir, csv_path)

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

		if check_job_status(tag, job_dir, jmax, args.spin_type, info_str, summary_rows):
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
