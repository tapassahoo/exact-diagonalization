# cli.py

import argparse
import os
import sys
from termcolor import colored

from pkg_utils.utils import whoami
from pkg_utils.config import *
from monomer_linear_rotor.utils import (
	convert_dipole_field_energy_to_cm_inv  # or appropriate import
)


# Define known molecular parameters
MOLECULE_DATA = {
	"HF":  {"dipole_moment": 1.83, "B_const": 20.559},
	"HCl": {"dipole_moment": 1.03, "B_const": 10.44},
	"HBr": {"dipole_moment": 0.78, "B_const": 8.467},
	"HI": {"dipole_moment": 0.38, "B_const": 6.51},
	"CO":  {"dipole_moment": 0.112, "B_const": 1.9225},
}

def parse_arguments():
	"""
	Parse command-line arguments for linear rotor simulation.
	"""
	parser = argparse.ArgumentParser(
		prog="monomer_rotor_real_basis_diagonalization.py",
		description=(
			"Exact diagonalization of the Hamiltonian for a linear polar rotor in an external electric field.\n\n"
			"Examples:\n"
			"  # Example 1: Use molecular name to auto-fill dipole moment and B constant\n"
			"  python monomer_rotor_real_basis_diagonalization.py 12 spinless \\\n"
			"		 --molecule HF \\\n"
			"		 --electric-field 100 \\\n"
			"		 --output-dir output/HF-spinless/\n\n"
			"  # Example 2: Specify dipole moment and potential strength manually\n"
			"  python monomer_rotor_real_basis_diagonalization.py 8 para \\\n"
			"		 --dipole-moment 1.5 \\\n"
			"		 --potential-strength 2.3 \\\n"
			"		 --output-dir output/custom_run/"
		),
		epilog="Developed by Dr. Tapas Sahoo — Quantum Molecular Dynamics Group",
		formatter_class=argparse.RawTextHelpFormatter
	)


	parser.add_argument("max_angular_momentum_quantum_number", type=int,
						help="Maximum angular momentum quantum number ℓ_max for basis truncation.")

	parser.add_argument("spin", choices=["spinless", "ortho", "para"],
						help="Spin isomer type: 'spinless', 'ortho', or 'para'.")

	parser.add_argument("--molecule", type=str,
						help="Molecule name (e.g., 'HF', 'HCl') to auto-fill μ and B.")

	parser.add_argument("--dipole-moment", type=float, default=None,
						help="Dipole moment in Debye (optional if --molecule is given).")

	parser.add_argument("--B-const", type=float, default=None,
						help="Rotational constant B (in cm⁻¹; optional if --molecule is given).")

	parser.add_argument("--electric-field", type=float, default=None,
						help="Electric field strength in kV/cm.")

	parser.add_argument("--potential-strength", type=float, default=None,
						help="V(θ) in cm⁻¹ (overrides μ·E if provided).")

	parser.add_argument("--output-dir", type=str, default="output",
						help="Output directory (default: 'output').")

	parser.add_argument("--dry-run", action="store_true",
						help="Print parameters and exit without running simulation.")

	args = parser.parse_args()

	# Auto-fill dipole and B from molecule name
	if args.molecule:
		mol = args.molecule.strip()
		if mol not in MOLECULE_DATA:
			print(colored(f"[Error] Molecule '{mol}' is not recognized.", "red"))
			sys.exit(1)
		if args.dipole_moment is None:
			args.dipole_moment = MOLECULE_DATA[mol]["dipole_moment"]
		if args.B_const is None:
			args.B_const = MOLECULE_DATA[mol]["B_const"]

	# Mandatory B constant
	if args.B_const is None:
		print(colored("[Error] Missing B constant. Use --B-const-cm-inv or --molecule.", "red"))
		sys.exit(1)

	# Compute V(θ) if not provided
	if args.potential_strength is None:
		if args.dipole_moment is not None and args.electric_field is not None:
			args.potential_strength = convert_dipole_field_energy_to_cm_inv(
				args.dipole_moment, args.electric_field
			)
		else:
			print(colored("[Error] Provide --potential-strength or both --dipole-moment and --electric-field.", "red"))
			sys.exit(1)

	return args


def show_dry_run_summary(args):
	"""
	Display a formatted summary of user inputs.
	"""
	print(colored("=" * (LABEL_WIDTH + VALUE_WIDTH), SEPARATOR_COLOR))
	print(colored("Dry Run Summary".center(LABEL_WIDTH + VALUE_WIDTH), HEADER_COLOR))
	print(colored("=" * (LABEL_WIDTH + VALUE_WIDTH), SEPARATOR_COLOR))

	entries = [
		("ℓ_max", args.max_angular_momentum_quantum_number),
		("Spin Isomer", args.spin),
		("Molecule", args.molecule or "N/A"),
		("B Constant", f"{args.B_const:.4f} cm⁻¹"),
		("Dipole Moment", f"{args.dipole_moment:.3f} D" if args.dipole_moment else "N/A"),
		("Electric Field", f"{args.electric_field:.3f} kV/cm" if args.electric_field else "N/A"),
		("Potential V(θ)", f"{args.potential_strength:.4f} cm⁻¹"),
		("Output Directory", args.output_dir),
	]

	for label, value in entries:
		print(colored(label.ljust(LABEL_WIDTH), LABEL_COLOR) +
			  colored(str(value).ljust(VALUE_WIDTH), VALUE_COLOR))

	print(colored("=" * (LABEL_WIDTH + VALUE_WIDTH), SEPARATOR_COLOR))


# Optional direct execution
if __name__ == "__main__":
	args = parse_arguments()

	if args.dry_run:
		show_dry_run_summary(args)
		sys.exit(0)

	os.makedirs(args.output_dir, exist_ok=True)

