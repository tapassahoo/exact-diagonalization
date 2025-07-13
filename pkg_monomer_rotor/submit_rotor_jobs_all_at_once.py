import argparse
import sys
import os
import shutil
import logging
import csv
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import List
from pkg_utils.utils import whoami
from pkg_utils.env_report import whom

MOLECULE_DATA = {
	"HF": {"dipole_moment": 1.83},
	"HCl": {"dipole_moment": 1.03},
	"HBr": {"dipole_moment": 0.78},
}

ALLOWED_SPIN_TYPES = ["spinless", "ortho", "para"]

def parse_arguments():
	parser = argparse.ArgumentParser(
		description=(
			"Submit rotor simulations for various angular momentum cutoffs and field strengths.\n\n"
			"Examples:\n"
			"  python submit_rotor_jobs_all_at_once.py ortho --molecule HF\n"
			"  python submit_rotor_jobs_all_at_once.py para --dipole-moment 1.75 --electric-fields 50 100 --jmax-values 6 8 10\n"
			"  python submit_rotor_jobs_all_at_once.py spinless --molecule HCl --use-potential --electric-fields 10 20 30\n"
		),
		epilog="Developed by Dr. Tapas Sahoo - Quantum Molecular Dynamics Group",
		formatter_class=argparse.RawTextHelpFormatter
	)

	parser.add_argument("spin_type", choices=ALLOWED_SPIN_TYPES,
						help="Spin isomer type: spinless, ortho, or para.")

	parser.add_argument("--molecule", type=str,
						help="Molecule name (e.g., HF, HCl) to auto-fill dipole moment (in Debye).")

	parser.add_argument("--dipole-moment", type=float, default=None,
						help="Dipole moment in Debye. Overrides value from --molecule if specified.")

	parser.add_argument("--electric-fields", type=float, nargs='+', default=[100.0],
						help="Electric field strengths (kV/cm) or potential strengths (cm^-1) if --use-potential is set.")

	parser.add_argument("--jmax-values", type=int, nargs='+', default=[6, 8, 10],
						help="List of maximum angular momentum quantum numbers to use.")

	parser.add_argument("--use-potential", action="store_true",
						help="Interpret electric-fields as potential strengths in cm^-1.")

	parser.add_argument("--dry-run", action="store_true",
						help="Only display the commands without executing them.")

	return parser.parse_args()

def setup_logging(log_file: str):
	os.makedirs(os.path.dirname(log_file), exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		handlers=[
			logging.FileHandler(log_file, mode='w'),
			logging.StreamHandler()
		]
	)

def build_command(jmax, value, use_dipole, dipole, output_dir, spin_type, script_name="main.py") -> List[str]:
	cmd = ["python3", script_name, str(jmax), spin_type, "--output-dir", output_dir]
	if use_dipole:
		cmd += ["--dipole-moment", str(dipole), "--electric-field", str(value)]
	else:
		cmd += ["--potential-strength", str(value)]
	return cmd

def generate_sh_file(cmd: List[str], job_dir: str, tag: str):
	run_script_path = os.path.join(job_dir, f"{tag}_run_command.sh")
	with open(run_script_path, "w") as f:
		f.write("#!/bin/bash\n")
		f.write(" ".join(cmd) + "\n")
	os.chmod(run_script_path, 0o755)

def write_summary_csv(rows: List[dict], csv_path: str):
	fieldnames = ["Job Name", "Max Angular Momentum", "Spin Type", "Field/Interaction", "Status", "PID"]
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def main():
	args = parse_arguments()

	# Determine dipole moment
	if not args.use_potential:
		if args.molecule:
			mol = args.molecule.strip().upper()
			if mol not in MOLECULE_DATA:
				print(f"[ERROR] Molecule '{mol}' not recognized. Choose from: {', '.join(MOLECULE_DATA)}")
				sys.exit(1)
			if args.dipole_moment is None:
				args.dipole_moment = MOLECULE_DATA[mol]["dipole_moment"]
		if args.dipole_moment is None:
			print("[ERROR] Either --molecule or --dipole-moment must be provided.")
			sys.exit(1)

	spin = args.spin_type
	molecule_tag = args.molecule if args.molecule else "custom"
	use_dipole = not args.use_potential
	script_name = "main.py"

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_dir = f"jobs_{molecule_tag}_{spin}_{timestamp}"
	os.makedirs(output_dir, exist_ok=True)

	log_file = os.path.join(output_dir, "submission.log")
	csv_path = os.path.join(output_dir, "job_summary.csv")
	setup_logging(log_file)

	summary = []

	for jmax, value in product(args.jmax_values, args.electric_fields):
		tag = f"{molecule_tag}_{spin}_J{jmax}_{'E' if use_dipole else 'V'}_{value}".replace(".", "_")
		job_dir = os.path.join(output_dir, tag)
		os.makedirs(job_dir, exist_ok=True)

		cmd = build_command(jmax, value, use_dipole, args.dipole_moment, job_dir, spin, script_name)
		generate_sh_file(cmd, job_dir, tag)

		status = "DRY-RUN" if args.dry_run else "SUBMITTED"

		if args.dry_run:
			print(f"[DRY-RUN] {tag}")
			print("  Command:", " ".join(cmd))
			print("  Script :", os.path.join(job_dir, f"{tag}_run_command.sh"))
		else:
			subprocess.Popen(cmd)

		summary.append({
			"Job Name": tag,
			"Max Angular Momentum": jmax,
			"Spin Type": spin,
			"Field/Interaction": f"{value} ({'kV/cm' if use_dipole else 'cm^-1'})",
			"Status": status,
			"PID": "-"
		})

	write_summary_csv(summary, csv_path)
	print("\nAll jobs processed.")
	print(f"Summary written to: {csv_path}")

if __name__ == "__main__":
	main()

