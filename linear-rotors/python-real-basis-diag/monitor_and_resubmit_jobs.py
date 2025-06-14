#!/usr/bin/env python3

import os
import subprocess
import csv
import glob

def detect_completion(stdout_path):
	"""Determine if a job appears completed based on output."""
	if not os.path.exists(stdout_path):
		return False
	with open(stdout_path, "r") as f:
		lines = f.readlines()
		if not lines:
			return False
		# Customize the completion indicator as needed
		for line in reversed(lines[-20:]):
			if "Completed" in line or "success" in line.lower():
				return True
	return False

def monitor_and_resubmit(base_output_dir):
	summary_csv_path = os.path.join(base_output_dir, "job_summary.csv")
	job_dirs = sorted([d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))])
	summary_rows = []

	print("Scanning job directories under:", base_output_dir)
	print()

	for job_tag in job_dirs:
		job_path = os.path.join(base_output_dir, job_tag)
		status_file = os.path.join(job_path, "job_status.txt")
		stdout_file = glob.glob(os.path.join(job_path, f"stdout_{job_tag}.txt"))
		run_script = os.path.join(job_path, "run_command.sh")

		stdout_path = stdout_file[0] if stdout_file else ""

		if os.path.exists(status_file):
			with open(status_file, "r") as f:
				current_status = f.read().strip()
		else:
			current_status = "UNKNOWN"

		if current_status == "COMPLETED":
			status = "COMPLETED"
		elif detect_completion(stdout_path):
			status = "COMPLETED"
		else:
			status = "FAILED"
			print(f"[{job_tag}] appears to have failed. Resubmitting...")

			# Resubmit the job
			if os.path.exists(run_script):
				subprocess.Popen(["bash", run_script], cwd=job_path)
				status = "RESUBMITTED"
			else:
				print(f"[{job_tag}] Missing run_command.sh. Cannot resubmit.")
				status = "FAILED"

		# Update status file
		with open(status_file, "w") as f:
			f.write(status + "\n")

		# Extract parameters from tag
		try:
			v0_str, j_str, spin = job_tag.split("_")
			v0 = float(v0_str[1:])
			jmax = int(j_str[1:])
		except Exception:
			v0 = None
			jmax = None
			spin = "unknown"

		summary_rows.append({
			"Job Name": job_tag,
			"Potential Strength": v0,
			"Max Angular Momentum": jmax,
			"Spin Type": spin,
			"Status": status
		})

	# Write updated summary CSV
	with open(summary_csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["Job Name", "Potential Strength", "Max Angular Momentum", "Spin Type", "Status"])
		writer.writeheader()
		writer.writerows(summary_rows)

	print()
	print("Monitoring complete. Updated job statuses saved to:")
	print(summary_csv_path)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Monitor and resubmit HF rotor jobs.")
	parser.add_argument("base_output_dir", help="Base output directory to monitor")
	args = parser.parse_args()

	monitor_and_resubmit(args.base_output_dir)

