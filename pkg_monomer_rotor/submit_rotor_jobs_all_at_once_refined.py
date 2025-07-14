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
import stat
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


MOLECULE_DATA = {
	"HF": {"dipole_moment": 1.83},
	"HCl": {"dipole_moment": 1.03},
	"HBr": {"dipole_moment": 0.78},
	"HI":  {"dipole_moment": 0.38},
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
			mol = args.molecule.strip()
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
		filename=log_file,
		filemode='w'
	)
	# Also print to console
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	console.setFormatter(formatter)
	logging.getLogger().addHandler(console)

def get_parameter_combinations(jmax_values, dipole_moment, electric_field_values, potential_strength_values):
	use_dipole_field = dipole_moment is not None and electric_field_values
	if use_dipole_field:
		return list(product(jmax_values, electric_field_values)), True
	else:
		return list(product(jmax_values, potential_strength_values)), False


def preview_job_settings(molecule, spin_type, dipole_moment, param_combinations, output_dir, csv_path, script_name):
	logging.info("=" * 50)
	logging.info(f"Launching job submissions for: {molecule}")
	logging.info("=" * 50)
	logging.info(f"{'Molecule':24}: {molecule}")
	logging.info(f"{'Spin Type':24}: {spin_type}")
	logging.info(f"{'Dipole Moment (D)':24}: {dipole_moment}")
	logging.info(f"{'Execution Script':24}: {script_name}")
	logging.info(f"{'Job Combinations':24}: {len(param_combinations)}")
	logging.info(f"{'Base Output Dir':24}: {output_dir}")
	logging.info(f"{'Summary CSV Path':24}: {csv_path}")
	logging.info("=" * 50)
	print()


def write_summary_csv(rows, csv_path):
	fieldnames = ["Job Name", "Max Angular Momentum", "Spin Type", "Field/Interaction", "Status", "PID"]
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def build_job_command(
	jmax,
	spin_type,
	molecule,
	electric_field=None,
	potential_strength=None,
	dipole_moment=None,
	dry_run=False,
	output_dir=None
):
	"""
	Build the command to execute the rotor diagonalization script.

	Parameters:
		jmax (int): Maximum angular momentum quantum number.
		spin_type (str): Spin type (e.g., 'singlet', 'triplet').
		molecule (str): Molecule name or tag.
		electric_field (float, optional): Field strength in kV/cm.
		potential_strength (float, optional): Interaction potential in cm⁻¹.
		dipole_moment (float, optional): Dipole moment in Debye.
		dry_run (bool): If True, appends the --dry-run flag.
		output_dir (str, optional): Path to output directory.

	Returns:
		list[str]: Command list to be passed to subprocess.
	"""
	cmd = [
		"python3", "main.py",
		str(jmax),
		str(spin_type),
		"--molecule", str(molecule)
	]

	if electric_field is not None:
		cmd += ["--electric-field", str(electric_field)]

	if potential_strength is not None:
		cmd += ["--potential-strength", str(potential_strength)]

	if dipole_moment is not None:
		cmd += ["--dipole-moment", str(dipole_moment)]

	if dry_run:
		cmd.append("--dry-run")

	if output_dir is not None:
		cmd += ["--output-dir", str(output_dir)]

	return cmd

def construct_job_tag(jmax, secondary_param, args, use_dipole):
	if use_dipole:
		return f"{args.spin_type}_{args.molecule}_jmax_{jmax}_field_{secondary_param:.2f}kV_per_cm"
	else:
		return f"{args.molecule}_Vfield_{secondary_param:.2f}cm1_Jmax_{jmax}_{args.spin_type}".replace(".", "_")

def construct_command(jmax, secondary_param, args, use_dipole, output_dir, script_path):
	if use_dipole:
		return [
			"python3", script_path,
			str(jmax),
			args.spin_type,
			"--electric-field", str(secondary_param),
			"--output-dir", output_dir
		]
	else:
		return [
			"python3", script_path,
			str(secondary_param),
			str(jmax),
			args.spin_type,
			"--output-dir", output_dir
		]

def prepare_job_directory(output_root_dir, tag, script_basename, script_source_path):
	output_dir = os.path.join(output_root_dir, tag)
	os.makedirs(output_dir, exist_ok=True)
	dest_script_path = os.path.join(output_dir, script_basename)
	shutil.copy2(script_source_path, dest_script_path)
	return output_dir, dest_script_path

def log_and_append_summary(summary_rows, job_status, tag, jmax, args, info_str):
	summary_rows.append({
		"Job Name": tag,
		"Max Angular Momentum": jmax,
		"Spin Type": args.spin_type,
		"Field/Interaction": info_str,
		"Status": job_status,
		"PID": "-"
	})

def submit_single_job(jmax, secondary_param, args, use_dipole, output_root_dir, script_basename, script_source_path, summary_rows):
	tag = construct_job_tag(jmax, secondary_param, args, use_dipole)
	info_str = (
		f"Dipole moment = {args.dipole_moment} D, E = {secondary_param} kV/cm"
		if use_dipole else
		f"V = {secondary_param} cm⁻¹"
	)

	output_dir, script_path = prepare_job_directory(output_root_dir, tag, script_basename, script_source_path)

	stdout_path = os.path.join(output_dir, f"{tag}.stdout")
	stderr_path = os.path.join(output_dir, f"{tag}.stderr")
	status_file = os.path.join(output_dir, "status.txt")

	status = determine_job_status(stdout_path, status_file)
	if status in ["COMPLETED", "RUNNING"]:
		print(f"[Skipping ] {tag}: status = {status}")
		log_and_append_summary(summary_rows, status, tag, jmax, args, info_str)
		return

	cmd = construct_command(jmax, secondary_param, args, use_dipole, output_dir, script_path)
	cmd_str = " ".join(cmd)

	if args.dry_run:
		print(f"\n[DRY RUN ] Job Name   : {tag}")
		print(f"			Command   : {cmd_str}")
		print(f"			Stdout	: {stdout_path}")
		print(f"			Stderr	: {stderr_path}")
		job_status = "DRY-RUN"
	else:
		print(f"\n[Launching] Job: {tag}")
		print(f"  > Command   : {cmd_str}")
		print(f"  > Stdout	: {stdout_path}")
		print(f"  > Stderr	: {stderr_path}")

		run_script_path = os.path.join(output_dir, f"{tag}_run_command.sh")
		with open(run_script_path, "w") as f:
			f.write("#!/bin/bash\n")
			f.write(cmd_str + "\n")

		with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
			subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)

		job_status = "SUBMITTED"

	log_and_append_summary(summary_rows, job_status, tag, jmax, args, info_str)

if False:
	def check_job_status(tag, job_dir, jmax, spin_type, info_str, summary_rows, mark_submitted=False):
		"""
		Checks the status of a job by inspecting status.txt, stdout, and stderr.
		If mark_submitted=True, writes status.txt = SUBMITTED if job is being launched.
		Returns True if job should be skipped, False otherwise.
		"""
		stdout_path = os.path.join(job_dir, f"{tag}.stdout")
		stderr_path = os.path.join(job_dir, f"{tag}.stderr")
		status_file = os.path.join(job_dir, "status.txt")

		# --- Check if job already completed based on stdout content ---
		job_completed = False
		if os.path.exists(stdout_path):
			with open(stdout_path, "r") as f:
				if "HURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES" in f.read():
					job_completed = True

		# --- Read current status if available ---
		current_status = None
		if os.path.exists(status_file):
			with open(status_file, "r") as f:
				current_status = f.read().strip().upper()

			# Remove PENDING status if nothing has started
			if current_status == "PENDING" and not os.path.exists(stdout_path):
				logging.warning(f"Removing stale status file for '{tag}' (PENDING but no stdout)")
				os.remove(status_file)
				current_status = None

		# --- Detect known runtime failures from stderr ---
		job_failed = False
		if os.path.exists(stderr_path):
			with open(stderr_path, "r") as f:
				err_content = f.read().lower()
				if any(k in err_content for k in ["error", "traceback", "segmentation fault"]):
					job_failed = True

		# --- Determine final status ---
		if job_completed or current_status == "COMPLETED":
			final_status = "COMPLETED"
		elif current_status == "RUNNING":
			final_status = "RUNNING"
		elif job_failed:
			final_status = "FAILED"
		else:
			final_status = "PENDING"
			if mark_submitted:
				with open(status_file, "w") as f:
					f.write("SUBMITTED")
				final_status = "SUBMITTED"
				#logging.info(f"[Submitted] Job status marked SUBMITTED for: {tag}")

		logging.info(f"[Status] {tag}: {final_status}")

		# Append status to summary table
		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": spin_type,
			"Field/Interaction": info_str,
			"Status": final_status,
			"PID": "-"
		})

		return final_status in {"COMPLETED", "RUNNING"}


def check_job_status(tag, job_dir, jmax, spin_type, info_str, summary_rows, dry_run=False):
    """
    Check the status of a job. Append status to summary_rows if the job is already submitted,
    completed, failed, or dry-run. Return True if the job should be skipped.
    """

    stdout_path = os.path.join(job_dir, f"{tag}.stdout")
    stderr_path = os.path.join(job_dir, f"{tag}.stderr")
    status_file = os.path.join(job_dir, "status.txt")

    # --- Dry-run: Report without execution ---
    if dry_run:
        logging.info(f"[DRY-RUN ] Ready to submit: {tag}")
        summary_rows.append({
            "Job Name": tag,
            "Max Angular Momentum": jmax,
            "Spin Type": spin_type,
            "Field/Interaction": info_str,
            "Status": "DRY-RUN",
            "PID": "-"
        })
        return True

    # --- Check for successful completion ---
    job_completed = False
    if os.path.exists(stdout_path):
        try:
            with open(stdout_path, "r") as f:
                content = f.read()
                job_completed = "HURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES" in content
        except Exception as e:
            logging.warning(f"[{tag}] Could not read stdout: {e}")

    # --- Read status file ---
    current_status = None
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                current_status = f.read().strip().upper()
        except Exception as e:
            logging.warning(f"[{tag}] Could not read status.txt: {e}")

    # --- Check for critical errors in stderr ---
    job_failed = False
    if os.path.exists(stderr_path):
        try:
            with open(stderr_path, "r") as f:
                content = f.read().lower()
                job_failed = any(keyword in content for keyword in ["error", "traceback", "segmentation fault"])
        except Exception as e:
            logging.warning(f"[{tag}] Could not read stderr: {e}")

    # --- Final status decision ---
    if job_completed or current_status == "COMPLETED":
        final_status = "COMPLETED"
    elif current_status == "RUNNING":
        final_status = "RUNNING"
    elif job_failed:
        final_status = "FAILED"
    else:
        return False  # Skip writing summary for fresh, unsubmitted jobs

    # --- Record the known status ---
    logging.info(f"[Status] {tag}: {final_status}")
    summary_rows.append({
        "Job Name": tag,
        "Max Angular Momentum": jmax,
        "Spin Type": spin_type,
        "Field/Interaction": info_str,
        "Status": final_status,
        "PID": "-"
    })

    return True  # Skip this job; it's not fresh

def main():
	args = parse_arguments()

	# Configuration
	script_name = "main.py"
	jmax_values = list(range(10, 21, 2))
	electric_field_values = [0.1] + list(range(20, 201, 20))
	potential_strength_values = [0.1, 0.5]
	use_dipole = True
	molecule_tag = args.molecule or "custom"

	# Execution banner
	mode = "DRY RUN" if args.dry_run else "EXECUTION"
	print(f"[{mode}] Submitting jobs for: molecule = {molecule_tag}, spin = {args.spin_type}\n")

	# Output setup
	output_root_dir = "output"
	os.makedirs(output_root_dir, exist_ok=True)
	script_basename = os.path.basename(script_name)

	csv_path = os.path.join(output_root_dir, f"job_summary_{args.spin_type}_{args.molecule}.csv")
	log_file = os.path.join(output_root_dir, f"job_submission_{args.spin_type}_{args.molecule}.log")
	setup_logging(log_file)

	# Generate parameter combinations
	param_combinations, use_dipole = get_parameter_combinations(
		jmax_values, args.dipole_moment, electric_field_values, potential_strength_values
	)

	preview_job_settings(
		molecule=args.molecule,
		spin_type=args.spin_type,
		dipole_moment=args.dipole_moment,
		param_combinations=param_combinations,
		output_dir=output_root_dir,
		csv_path=csv_path,
		script_name=script_name
	)

	summary_rows = []

	# Main loop over job combinations
	for jmax, param in param_combinations:
		if use_dipole:
			E = param
			tag = f"{args.spin_type}_{args.molecule}_jmax_{jmax}_field_{E:.2f}kV_per_cm"
			info_str = f"Dipole moment = {args.dipole_moment} D, E = {E:.2f} kV/cm"
			cmd = build_job_command(
				jmax=jmax,
				spin_type=args.spin_type,
				molecule=args.molecule,
				electric_field=E,
				dry_run=args.dry_run,
				#output_dir=os.path.join(output_root_dir, tag)
			)
		else:
			V = param
			tag = f"{args.molecule}_Vfield_{V:.2f}cm1_Jmax_{jmax}_{args.spin_type}".replace(".", "_")
			info_str = f"V = {V:.2f} cm⁻¹"
			cmd = build_job_command(
				jmax=jmax,
				spin_type=args.spin_type,
				molecule=args.molecule,
				potential_strength=V,
				dry_run=args.dry_run,
				#output_dir=os.path.join(output_root_dir, tag)
			)

		job_dir = os.path.join(output_root_dir, tag)
		os.makedirs(job_dir, exist_ok=True)
		shutil.copy2(script_name, os.path.join(job_dir, script_basename))

		#skip = check_job_status(tag, job_dir, jmax, args.spin_type, info_str, summary_rows, mark_submitted=not args.dry_run)
		skip_job = check_job_status(
			tag=tag,
			job_dir=job_dir,
			jmax=jmax,
			spin_type=args.spin_type,
			info_str=info_str,
			summary_rows=summary_rows,
			dry_run=args.dry_run
		)

		if skip_job:
			continue

		# Prepare I/O paths
		stdout_path = os.path.join(job_dir, f"{tag}.stdout")
		stderr_path = os.path.join(job_dir, f"{tag}.stderr")
		cmd_str = " ".join(cmd)

		if args.dry_run:
			print(f"\n[DRY RUN ] Job Name: {tag}")
			print(f"			Command: {cmd_str}")
			print(f"			Stdout : {stdout_path}")
			print(f"			Stderr : {stderr_path}")
			status = "DRY-RUN"
		else:
			print(f"\n[Launching] Job: {tag}")
			print(f"  > Command : {cmd_str}")
			print(f"  > Stdout  : {stdout_path}")
			print(f"  > Stderr  : {stderr_path}\n\n")

			# Write shell script
			with open(os.path.join(job_dir, f"{tag}_run_command.sh"), "w") as f:
				f.write("#!/bin/bash\n" + cmd_str + "\n")

			# Launch process
			with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
				subprocess.Popen(cmd, stdout=out_f, stderr=err_f)

			"""
			# --- Create shell script ---
			run_script_path = os.path.join(job_dir, f"{tag}_run_command.sh")
			with open(run_script_path, "w") as f:
				f.write("#!/bin/bash\n")
				f.write(cmd_str + "\n")

			# --- Make the script executable ---
			st = os.stat(run_script_path)
			os.chmod(run_script_path, st.st_mode | stat.S_IEXEC)

			# --- Launch the shell script ---
			with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
				subprocess.Popen(["bash", run_script_path], stdout=out_f, stderr=err_f)
			"""

			status = "SUBMITTED"

		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": args.spin_type,
			"Field/Interaction": info_str,
			"Status": status,
			"PID": "-"
		})

	# Final summary
	write_summary_csv(summary_rows, csv_path)
	print()
	logging.info("All jobs processed. Summary written to CSV.")

	print("==========================================")
	print("HURRAY! ALL JOBS SUBMITTED SUCCESSFULLY")
	print(f"[ ] Output directory : {output_root_dir}")
	print(f"[ ] Summary CSV      : {csv_path}")
	print("==========================================")

# =============================
# Place this at the very end
# =============================
if __name__ == "__main__":
	main()

