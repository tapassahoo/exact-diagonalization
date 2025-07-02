import os
import subprocess
import csv
from itertools import product
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom


# -----------------------------
# CONFIGURATION
# -----------------------------
# Case A: dipole + field is provided
#dipole_moment_D = 1.827
dipole_moment_D = 1.093
electric_field_kVcm_list = [0.1] + list(range(10, 201, 10))

# Case B (ignored if A is active): V-field values
potential_strength_list = [0.1, 0.5]

max_angular_momentum_list = list(range(10, 31, 5))
spin_type = "spinless"  # Choose from: "spinless", "ortho", "para"

script_name = "monomer_rotor_real_basis_diagonalization.py"

# -----------------------------
# CHOOSE JOB PARAMETER MODE
# -----------------------------
base_output_dir = f"output_{spin_type}_HCl_monomer_in_field"
use_dipole_field = dipole_moment_D is not None and electric_field_kVcm_list

if use_dipole_field:
	param_combinations = list(product(max_angular_momentum_list, electric_field_kVcm_list))
else:
	param_combinations = list(product(max_angular_momentum_list, potential_strength_list))

os.makedirs(base_output_dir, exist_ok=True)
summary_csv_path = os.path.join(base_output_dir, "job_summary.csv")

# -----------------------------
# PREVIEW
# -----------------------------
print("==========================================")
print("Launching HCl monomer rotor job submissions")
print("==========================================")
print(f"Spin type              : {spin_type}")
print(f"Script to execute      : {script_name}")
print(f"Total job combinations : {len(param_combinations)}")
print(f"Base output directory  : {base_output_dir}")
print(f"Summary CSV will be at : {summary_csv_path}")
print("------------------------------------------\n")

summary_rows = []

# -----------------------------
# JOB LOOP
# -----------------------------
for jmax, secondary_param in param_combinations:
	if use_dipole_field:
		E = secondary_param
		tag = f"{spin_type}_HCl_lmax_{jmax}_dipole_moment_{dipole_moment_D:.2f}D_electric_field_{E:.2f}kVcm".replace(".", "_")
		cmd = [
			"python3", script_name,
			str(jmax),
			spin_type,
			"--dipole-moment", str(dipole_moment_D),
			"--electric-field", str(E),
			"--output-dir", os.path.join(base_output_dir, tag)
		]
		info_str = f"μ={dipole_moment_D} D, E={E} kV/cm"
	else:
		V = secondary_param
		tag = f"HCl_Vfield{V:.2f}cm1_Jmax{jmax}_{spin_type}".replace(".", "_")
		cmd = [
			"python3", script_name,
			str(V), str(jmax), spin_type,
			"--output-dir", os.path.join(base_output_dir, tag)
		]
		info_str = f"V={V} cm⁻¹"

	job_dir = os.path.join(base_output_dir, tag)
	os.makedirs(job_dir, exist_ok=True)

	stdout_path = os.path.join(job_dir, f"stdout_{tag}.txt")
	stderr_path = os.path.join(job_dir, f"stderr_{tag}.txt")
	status_file = os.path.join(job_dir, "job_status.txt")

	job_completed = False
	if os.path.exists(stdout_path):
		with open(stdout_path, "r") as f:
			contents = f.read()
			if "HURRAY ALL COMPUTATIONS COMPLETED DATA SUCCESSFULLY WRITTEN TO NETCDF FILES" in contents:
				job_completed = True

	if os.path.exists(status_file):
		with open(status_file) as f:
			current_status = f.read().strip().upper()
		if current_status == "COMPLETED" or job_completed:
			print(f"[Skipping ] {tag}: status = COMPLETED")
			summary_rows.append({
				"Job Name": tag,
				"Max Angular Momentum": jmax,
				"Spin Type": spin_type,
				"Field/Interaction": info_str,
				"Status": "COMPLETED",
				"PID": "-"
			})
			continue
		elif current_status == "RUNNING":
			print(f"[Skipping ] {tag}: status = RUNNING")
			summary_rows.append({
				"Job Name": tag,
				"Max Angular Momentum": jmax,
				"Spin Type": spin_type,
				"Field/Interaction": info_str,
				"Status": "RUNNING",
				"PID": "-"
			})
			continue

	cmd_str = " ".join(cmd)
	print(f"[Launching] Job: {tag}")
	print(f"  > Command	  : {cmd_str}")

	run_script_path = os.path.join(job_dir, "run_command.sh")
	with open(run_script_path, "w") as f:
		f.write("#!/bin/bash\n")
		f.write(cmd_str + "\n")

	print(f"  > Stdout path  : {stdout_path}")
	print(f"  > Stderr path  : {stderr_path}")

	stdout_file = open(stdout_path, "w")
	stderr_file = open(stderr_path, "w")

	process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)
	pid = process.pid
	print(f"  > Process ID   : {pid}")
	print("  > Status	   : RUNNING\n")

	with open(os.path.join(job_dir, "pid.txt"), "w") as f:
		f.write(str(pid) + "\n")

	with open(status_file, "w") as f:
		f.write("RUNNING\n")

	summary_rows.append({
		"Job Name": tag,
		"Max Angular Momentum": jmax,
		"Spin Type": spin_type,
		"Field/Interaction": info_str,
		"Status": "RUNNING",
		"PID": pid
	})

# -----------------------------
# WRITE SUMMARY FILE
# -----------------------------
with open(summary_csv_path, "w", newline="") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=[
		"Job Name", "Max Angular Momentum", "Spin Type", "Field/Interaction", "Status", "PID"
	])
	writer.writeheader()
	writer.writerows(summary_rows)

# -----------------------------
# FINAL MESSAGE
# -----------------------------
print("==========================================")
print("HURRAY ALL JOBS SUBMITTED SUCCESSFULLY")
print(f"Output directory: {base_output_dir}")
print(f"Summary CSV: {summary_csv_path}")
print("==========================================")

