#!/usr/bin/env python3

import os
import subprocess
import csv
from itertools import product

# -----------------------------
# CONFIGURATION
# -----------------------------
potential_strength_list = [0.1, 0.5]
max_angular_momentum_list = [6, 8]
spin_type = "spinless"  # Choose from: "spinless", "ortho", "para"
script_name = "monomer_rotor_real_basis_diagonalization.py"
base_output_dir = f"output_{spin_type}_HF_monomer_in_field"

os.makedirs(base_output_dir, exist_ok=True)
summary_csv_path = os.path.join(base_output_dir, "job_summary.csv")

total_jobs = len(potential_strength_list) * len(max_angular_momentum_list)

print("==========================================")
print("Launching HF monomer rotor job submissions")
print("==========================================")
print(f"Spin type              : {spin_type}")
print(f"Script to execute      : {script_name}")
print(f"Total job combinations : {total_jobs}")
print(f"Base output directory  : {base_output_dir}")
print(f"Summary CSV will be at : {summary_csv_path}")
print("------------------------------------------\n")

summary_rows = []

# -----------------------------
# JOB LOOP
# -----------------------------
for v0, jmax in product(potential_strength_list, max_angular_momentum_list):
	tag = f"HF_Vfield{v0:.2f}kVcm_Jmax{jmax}_{spin_type}".replace(".", "_")
	job_dir = os.path.join(base_output_dir, tag)
	os.makedirs(job_dir, exist_ok=True)

	status_file = os.path.join(job_dir, "job_status.txt")
	if os.path.exists(status_file):
		with open(status_file) as f:
			current_status = f.read().strip().upper()
			if current_status in ["COMPLETED", "RUNNING"]:
				print(f"[Skipping ] {tag}: status = {current_status}")
				summary_rows.append({
					"Job Name": tag,
					"Potential Strength": v0,
					"Max Angular Momentum": jmax,
					"Spin Type": spin_type,
					"Status": current_status,
					"PID": "-"
				})
				continue

	cmd = [
		"python3", script_name,
		str(v0), str(jmax), spin_type,
		"--output-dir", job_dir
	]
	cmd_str = " ".join(cmd)

	print(f"[Launching] Job: {tag}")
	print(f"  > Command	 : {cmd_str}")

	run_script_path = os.path.join(job_dir, "run_command.sh")
	with open(run_script_path, "w") as f:
		f.write("#!/bin/bash\n")
		f.write(cmd_str + "\n")

	stdout_path = os.path.join(job_dir, f"stdout_{tag}.txt")
	stderr_path = os.path.join(job_dir, f"stderr_{tag}.txt")
	print(f"  > Stdout path  : {stdout_path}")
	print(f"  > Stderr path  : {stderr_path}")

	stdout_file = open(stdout_path, "w")
	stderr_file = open(stderr_path, "w")

	process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
	pid = process.pid
	print(f"  > Process ID   : {pid}")
	print("  > Status       : RUNNING\n")

	with open(os.path.join(job_dir, "pid.txt"), "w") as f:
		f.write(str(pid) + "\n")

	with open(status_file, "w") as f:
		f.write("RUNNING\n")

	summary_rows.append({
		"Job Name": tag,
		"Potential Strength": v0,
		"Max Angular Momentum": jmax,
		"Spin Type": spin_type,
		"Status": "RUNNING",
		"PID": pid
	})

# -----------------------------
# WRITE SUMMARY FILE
# -----------------------------
with open(summary_csv_path, "w", newline="") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=["Job Name", "Potential Strength", "Max Angular Momentum", "Spin Type", "Status", "PID"])
	writer.writeheader()
	writer.writerows(summary_rows)

# -----------------------------
# FINAL MESSAGE
# -----------------------------
print("==========================================")
print("Job submission complete.")
print(f"Output directory: {base_output_dir}")
print(f"Summary CSV: {summary_csv_path}")
print("==========================================")

