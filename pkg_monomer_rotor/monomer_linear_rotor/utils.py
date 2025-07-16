import os
import getpass
import numpy as np
from scipy.sparse import issparse
from datetime import datetime
from termcolor import colored
from typing import Optional
from pkg_utils.utils import whoami
from pkg_utils.config import *

def is_hermitian(H, tol=1e-12):
	"""
	Check whether a matrix is Hermitian (sparse or dense).
	"""
	if issparse(H):
		return (H - H.getH()).nnz == 0
	else:
		return np.allclose(H, H.conj().T, atol=tol)

def show_simulation_details(
	output_root_dir,
	B_const_cm_inv,
	potential_strength_cm_inv,
	max_angular_momentum_quantum_number,
	spin_state,
	dipole_moment_D=None,
	electric_field_kVcm=None,
	computed_muE_cm_inv=None
):
	"""
	Display simulation input details including dipole-field interaction info if available.
	"""
	now = datetime.now()
	date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
	user_name = getpass.getuser()
	cwd = os.getcwd()
	home_dir = os.path.expanduser("~")

	print(colored("*" * 80, SEPARATOR_COLOR))
	print(colored("[ ] Date and Time:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(date_time.ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")

	print(colored("File System Details:", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("[ ] User Name:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(user_name.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Home Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(home_dir.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Current Working Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(cwd.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Output will be saved to:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(output_root_dir.ljust(VALUE_WIDTH), VALUE_COLOR))
	print()

	print(colored("Simulation Parameters", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("[ ] ℓ_max (Angular Momentum):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{max_angular_momentum_quantum_number}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Spin State:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(spin_state.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"[ ] Rotational constant:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{B_const_cm_inv:.6f} cm^-1".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] V(θ) Strength:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{potential_strength_cm_inv:.5f} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))

	if dipole_moment_D is not None:
		print(colored("[ ] Dipole Moment:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{dipole_moment_D:.4f} D".ljust(VALUE_WIDTH), VALUE_COLOR))
	if electric_field_kVcm is not None:
		print(colored("[ ] Electric Field:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{electric_field_kVcm:.4f} kV/cm".ljust(VALUE_WIDTH), VALUE_COLOR))
	if computed_muE_cm_inv is not None:
		print(colored("[ ] μ·E (Interaction Energy):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{computed_muE_cm_inv:.5f} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))

def generate_filename(
	molecule_name: str,
	spin_state: str,
	max_angular_momentum_quantum_number: int,
	potential_strength: float,
	dipole_moment_D: Optional[float] = None,
	electric_field_kVcm: Optional[float] = None,
	prefix: Optional[str] = ""
) -> str:
	"""
	Generates a descriptive filename for a quantum rotor system.

	Parameters
	----------
	molecule_name : str
		Name of the linear rigid rotor (e.g., "HF", "HCl", "CO")
	spin_state : str
		The spin isomer type ("spinless", "para", or "ortho").
	max_angular_momentum_quantum_number : int
		Maximum angular momentum quantum number (ℓ_max).
	potential_strength : float
		Orienting potential strength in Kelvin (used only if dipole-field interaction is not specified).
	dipole_moment_D : float, optional
		Dipole moment in Debye (include only if electric_field_kVcm is also provided).
	electric_field_kVcm : float, optional
		Electric field strength in kV/cm (include only if dipole_moment_D is also provided).
	prefix : str, optional
		Optional prefix or directory path.

	Returns
	-------
	str
		A clear and descriptive filename.
	"""

	filename = (
		f"{prefix}_{spin_state}_{molecule_name}_"
		f"jmax_{max_angular_momentum_quantum_number}_"
	)

	if dipole_moment_D is not None and electric_field_kVcm is not None:
		filename += (
			#f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"field_{electric_field_kVcm:.2f}kV_per_cm"
		)
	else:
		filename += f"potential_{potential_strength:.2f}cm_inv"

	return filename

def display_eigenvalues(eigenvalues, spin_state, unit="cm^-1", precision=6):
	"""
	Display eigenvalues with indices and physical units.

	Parameters:
		eigenvalues (array-like): Array or list of eigenvalues.
		header (str): Descriptive header for the table.
		unit (str): Physical unit of eigenvalues. Default is 'cm^-1'.
		precision (int): Number of decimal places to display.
	"""
	if eigenvalues is None or len(eigenvalues) == 0:
		print("[Info] No eigenvalues to display.")
		return

	print(
		colored("\n[INFO]", INFO_COLOR) +
		colored(" Lowest energy eigenvalues for spin type ", LABEL_COLOR) +
		colored(f"{spin_state}:", VALUE_COLOR)
	)


	#print(f"\n[ ] Lowest energy eigenvalues for spin type '{spin_state}':")
	print("-" * 50)
	for idx, val in enumerate(eigenvalues):
		formatted_value = f"{val:.{precision}f}"
		print(f"  Level {idx:<3} : {formatted_value:>12} {unit}")
	print("-" * 50)
	print(
		colored("[INFO]", INFO_COLOR) + " " +
		colored("Total levels: ", LABEL_COLOR) +
		colored(f"{len(eigenvalues)}", VALUE_COLOR) +
		"\n"
	)


def convert_dipole_field_energy_to_cm_inv(dipole_moment_D: float, electric_field_kVcm: float) -> float:
	"""
	Conversion of Dipole–Electric Field Interaction Energy to Wavenumbers (cm⁻¹):

	The interaction energy between a dipole and an external electric field is given by:

		U = -μ · E

	Where:
		- μ is the dipole moment (in Debye),
		- E is the electric field (in kV/cm).

	To convert the product μ × E to energy units of cm⁻¹, we use the following steps:

	1. 1 Debye = 3.33564 × 10⁻³⁰ C·m
	2. 1 kV/cm = 10⁵ V/m
	3. 1 cm⁻¹ = 1.98630 × 10⁻²³ J (via E = hcν)

	Thus, the energy in cm⁻¹ becomes:

		(μ [D] × E [kV/cm]) × (3.33564×10⁻³⁰ × 10⁵) / (1.98630×10⁻²³)
	  = μ [D] × E [kV/cm] × 0.01679 cm⁻¹

	Therefore:

		μ × E (in cm⁻¹) = dipole_moment_D * electric_field_kVcm * 0.01679

	This factor (0.01679) accounts for conversion from (Debye × kV/cm) to cm⁻¹ using physical constants.

	Parameters:
		dipole_moment_D (float): Dipole moment in Debye.
		electric_field_kVcm (float): Electric field in kilovolts per centimeter.

	Returns:
		float: Interaction energy in cm⁻¹.
	"""
	return dipole_moment_D * electric_field_kVcm * 0.01679

