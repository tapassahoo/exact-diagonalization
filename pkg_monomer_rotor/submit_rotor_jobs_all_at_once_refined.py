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

width = 70
separator = "=" * width
bar = "=" * 60
bar = "=" * 72
label_width = 12

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

def get_parameter_combinations(jmax_values, dipole_moment, electric_field_values, potential_strength_values):
	use_dipole_field = dipole_moment is not None and electric_field_values
	if use_dipole_field:
		return list(product(jmax_values, electric_field_values)), True
	else:
		return list(product(jmax_values, potential_strength_values)), False


def preview_job_settings(
	molecule,
	spin_type,
	dipole_moment,
	param_combinations,
	output_dir,
	log_file,
	script_name
):
	"""
	Print a summary of job settings before execution.
	"""

	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	lines = [
		separator,
		f"{'Job Submission Summary'.center(width)}",
		separator,
		f"{'Timestamp':24}: {timestamp}",
		f"{'Molecule':24}: {molecule}",
		f"{'Spin Type':24}: {spin_type}",
		f"{'Dipole Moment (D)':24}: {dipole_moment}",
		f"{'Execution Script':24}: {script_name}",
		f"{'Job Combinations':24}: {len(param_combinations)}",
		f"{'Base Output Dir':24}: {output_dir}",
		f"{'Job Submission Summary':24}: {log_file}",
		separator,
		""
	]

	for line in lines:
		print(line)

def write_summary_txt(rows, txt_file):
	"""
	Write the job summary to a text file in tabular format.

	Parameters:
	- rows: List of dictionaries containing job details.
	- txt_file: Path to the output .txt file.
	"""
	fieldnames = ["Job Name", "Max Angular Momentum", "Spin Type", "Field/Interaction", "Status", "PID"]
	
	# Determine max width for each column
	col_widths = {field: len(field) for field in fieldnames}
	for row in rows:
		for field in fieldnames:
			col_widths[field] = max(col_widths[field], len(str(row[field])))

	with open(txt_file, "w") as f:
		# Header
		header = " | ".join(f"{field:<{col_widths[field]}}" for field in fieldnames)
		separator = "-+-".join("-" * col_widths[field] for field in fieldnames)
		f.write(header + "\n")
		f.write(separator + "\n")

		# Data rows
		for row in rows:
			line = " | ".join(f"{str(row[field]):<{col_widths[field]}}" for field in fieldnames)
			f.write(line + "\n")

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

	label_width = 18  # Width for labels

	if args.dry_run:
		print(f"\n[DRY-RUN   ] {'Job Name':<{label_width}}: {tag}")
		print(f"			  {'Command':<{label_width}}: {cmd_str}")
		print(f"			  {'Stdout Path':<{label_width}}: {stdout_path}")
		print(f"			  {'Stderr Path':<{label_width}}: {stderr_path}")
		job_status = "DRY-RUN"
	else:
		print(f"\n[LAUNCHING ] {'Job Name':<{label_width}}: {tag}")
		print(f"			  {'Command':<{label_width}}: {cmd_str}")
		print(f"			  {'Stdout Path':<{label_width}}: {stdout_path}")
		print(f"			  {'Stderr Path':<{label_width}}: {stderr_path}")

		run_script_path = os.path.join(output_dir, f"{tag}_run_command.sh")
		with open(run_script_path, "w") as f:
			f.write("#!/bin/bash\n")
			f.write(cmd_str + "\n")

		with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
			subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)

		job_status = "SUBMITTED"

	log_and_append_summary(summary_rows, job_status, tag, jmax, args, info_str)

def check_job_status(
	tag,
	job_dir,
	jmax,
	spin_type,
	info_str,
	summary_rows,
	dry_run=False,
	script_name=None,
	cmd=None
):
	"""
	Check the status of a job. Append status to summary_rows if the job is already submitted,
	completed, running, failed, or dry-run. Return True if the job should be skipped.
	"""

	stdout_path = os.path.join(job_dir, f"{tag}.stdout")
	stderr_path = os.path.join(job_dir, f"{tag}.stderr")
	cmd_str = " ".join(cmd)

	# --- DRY-RUN handling ---
	if dry_run:
		print(f"\n[DRY RUN]")
		print(f"{'':{label_width}}{'Job Name':<{label_width}}: {tag}")
		print(f"{'':{label_width}}{'Command':<{label_width}}: {cmd_str}")
		print(f"{'':{label_width}}{'Stdout':<{label_width}}: {stdout_path}")
		print(f"{'':{label_width}}{'Stderr':<{label_width}}: {stderr_path}")
		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": spin_type,
			"Field/Interaction": info_str,
			"Status": "DRY-RUN",
			"PID": "-"
		})
		return True

	# --- Create job_dir if it does not exist ---
	if not os.path.exists(job_dir):
		os.makedirs(job_dir, exist_ok=True)

		# Copy script if provided
		if script_name:
			try:
				shutil.copy2(script_name, os.path.join(job_dir, os.path.basename(script_name)))
			except Exception as e:
				print(f"[WARNING   ] {tag:<24}: Failed to copy script — {e}")

		# Save run command if provided
		if cmd_str:
			try:
				run_script = os.path.join(job_dir, f"{tag}_run_command.sh")
				with open(run_script, "w") as f:
					f.write("#!/bin/bash\n")
					f.write(f"{cmd_str}\n")
				os.chmod(run_script, 0o755)
				# Launch process
				with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
					subprocess.Popen(cmd, stdout=out_f, stderr=err_f)

			except Exception as e:
				print(f"[WARNING   ] {tag:<{label_width}}: Failed to write run_command.sh — {e}")
				return

		print(f"[NEW JOB] {tag:<{label_width}}: Directory created.")
		print(f"{'':{label_width}}{'Job Name':<{label_width}}: {tag}")
		print(f"{'':{label_width}}{'Command':<{label_width}}: {cmd_str}")
		print(f"{'':{label_width}}{'Stdout':<{label_width}}: {stdout_path}")
		print(f"{'':{label_width}}{'Stderr':<{label_width}}: {stderr_path}")
		print(f"[SUBMITTED] \n")

		return False  # Proceed to submit

	# --- Check stderr for critical errors ---
	job_failed = False
	stderr_error_message = ""
	if os.path.exists(stderr_path) and os.path.getsize(stderr_path) > 0:
		try:
			with open(stderr_path, "r") as f:
				content = f.read().lower()
				for keyword in ["error", "traceback", "segmentation fault"]:
					if keyword in content:
						stderr_error_message = f"Detected '{keyword}' in stderr"
						job_failed = True
						break
		except Exception as e:
			print(f"[WARNING   ] {tag:<{label_width}}: Could not read stderr — {e}")

	if job_failed:
		print(f"[FAILED	] {tag:<{label_width}}: {stderr_error_message}")
		print(f"			  Suggestion : Delete the directory and resubmit the job.")

		user_input = input(f">>> Delete directory '{job_dir}' and create resubmit.sh? [y/N]: ").strip().lower()
		if user_input == 'y':
			try:
				shutil.rmtree(job_dir)
				os.makedirs(job_dir, exist_ok=True)

				# Write resubmit.sh
				resubmit_path = os.path.join(job_dir, "resubmit.sh")
				with open(resubmit_path, "w") as f:
					f.write("#!/bin/bash\n")
					f.write("# Auto-generated resubmission script\n")
					f.write("bash run.sh\n")  # Update this if needed
				os.chmod(resubmit_path, 0o755)

				print(f"[CLEANUP   ] {tag:<24}: resubmit.sh created after cleanup.")
			except Exception as e:
				print(f"[ERROR	 ] {tag:<24}: Failed to reset directory — {e}")
		else:
			print(f"[SKIPPED   ] {tag:<24}: User chose not to delete the job directory.")

		summary_rows.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": spin_type,
			"Field/Interaction": info_str,
			"Status": "FAILED",
			"PID": "-"
		})
		return True

	# --- If stdout_path exists, skip this job ---
	if os.path.exists(stdout_path):
		try:
			with open(stdout_path, "r") as f:
				content = f.read()
				if "HURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES" in content:
					status = "COMPLETED"
					print(f"[{status} ] {tag:<{label_width}}")
					#print(f"{'':{label_width}}{'Command':<{label_width}}: {cmd_str}")
					#print(f"{'':{label_width}}{'Stdout':<{label_width}}: {stdout_path}")
					#print(f"{'':{label_width}}{'Stderr':<{label_width}}: {stderr_path}\n")
				elif len(content.strip()) > 0:
					status = "RUNNING"
					print(f"[{status} ] {tag:<{label_width}}: stdout indicates job is still running.")
					#print(f"{'':{label_width}}{'Command':<{label_width}}: {cmd_str}")
					print(f"{'':{label_width}}{'Stdout':<{label_width}}: {stdout_path}")
					#print(f"{'':{label_width}}{'Stderr':<{label_width}}: {stderr_path}\n")
				else:
					status = "PENDING"
					print(f"[{status} ] {tag:<{label_width}}: stdout exists but is empty.")
					#print(f"{'':{label_width}}{'Command':<{label_width}}: {cmd_str}")
					print(f"{'':{label_width}}{'Stdout':<{label_width}}: {stdout_path}")
					#print(f"{'':{label_width}}{'Stderr':<{label_width}}: {stderr_path}\n")
		except Exception as e:
			status = "UNKNOWN"
			print(f"[WARNING   ] {tag:<{label_width}}: Could not read stdout — {e}")
		else:
			summary_rows.append({
				"Job Name": tag,
				"Max Angular Momentum": jmax,
				"Spin Type": spin_type,
				"Field/Interaction": info_str,
				"Status": status,
				"PID": "-"
			})
			return True  # Skip job

	# --- If stdout does not exist, proceed to submission ---
	return False


def print_submission_success(output_root_dir, log_file):
	print(bar)
	print(f"{'HURRAY! ALL JOBS SUBMITTED SUCCESSFULLY':^{label_width}}")
	print(bar)
	print(f"{'Output Directory':<{label_width}}: {output_root_dir}")
	print(f"{'Summary CSV':<{label_width}}: {log_file}")
	print(bar)
	print()


def main():
	args = parse_arguments()

	# Configuration
	script_name = "main.py"
	jmax_values = list(range(20, 41, 10))
	electric_field_values = [0.1] + list(range(100, 501, 100))
	potential_strength_values = [0.1, 0.5]
	use_dipole = True
	molecule_tag = args.molecule or "custom"

	# Execution banner
	mode = "DRY RUN" if args.dry_run else "EXECUTION"
	print(f"[{mode}] Submitting jobs for: molecule = {molecule_tag}, spin = {args.spin_type}\n")

	# Output setup
	output_root_dir = "output"
	log_file = os.path.join(output_root_dir, f"job_submission_{args.spin_type}_{args.molecule}.txt")

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
		log_file=log_file,
		script_name=script_name,
	)

	if mode == "EXECUTION":
		os.makedirs(output_root_dir, exist_ok=True)

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
			)

		# Prepare I/O paths
		job_dir = os.path.join(output_root_dir, tag)

		skip_job=check_job_status(
			tag=tag,
			job_dir=job_dir,
			jmax=jmax,
			spin_type=args.spin_type,
			info_str=info_str,
			summary_rows=summary_rows,
			dry_run=args.dry_run,
			script_name=script_name,
			cmd=cmd
		)

		if skip_job:
			continue

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
	if mode == "EXECUTION": 
		write_summary_txt(summary_rows, log_file)
	print(f"\n\nAll jobs processed.")

		#print_submission_success(output_root_dir, log_file)

# =============================
# Place this at the very end
# =============================
if __name__ == "__main__":
	main()

