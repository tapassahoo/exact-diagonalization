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
	print(colored("Date and Time:".ljust(LABEL_WIDTH), INFO_COLOR) + colored(date_time.ljust(VALUE_WIDTH), VALUE_COLOR) + "\n")

	print(colored("File System Details:", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("User Name:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(user_name.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Home Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(home_dir.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Current Working Directory:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(cwd.ljust(VALUE_WIDTH), VALUE_COLOR))
	#print(colored("Package Location (bfunc):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(bfunc.__file__.ljust(VALUE_WIDTH), VALUE_COLOR))
	print()

	print(colored("Simulation Parameters", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("ℓ_max (Angular Momentum):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{max_angular_momentum_quantum_number}".ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("Spin State:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(spin_state.ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("V(θ) Strength:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{potential_strength_cm_inv:.5f} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))

	if dipole_moment_D is not None:
		print(colored("Dipole Moment:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{dipole_moment_D:.4f} D".ljust(VALUE_WIDTH), VALUE_COLOR))
	if electric_field_kVcm is not None:
		print(colored("Electric Field:".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{electric_field_kVcm:.4f} kV/cm".ljust(VALUE_WIDTH), VALUE_COLOR))
	if computed_muE_cm_inv is not None:
		print(colored("μ·E (Interaction Energy):".ljust(LABEL_WIDTH), LABEL_COLOR) + colored(f"{computed_muE_cm_inv:.5f} cm⁻¹".ljust(VALUE_WIDTH), VALUE_COLOR))
	print()

	print(colored("Grid Information", HEADER_COLOR, attrs=['bold', 'underline']))

def generate_filename(
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
		f"{prefix}HCl_{spin_state}_isomer_"
		f"lmax_{max_angular_momentum_quantum_number}_"
	)

	if dipole_moment_D is not None and electric_field_kVcm is not None:
		filename += (
			f"dipole_moment_{dipole_moment_D:.2f}D_"
			f"electric_field_{electric_field_kVcm:.2f}kVcm"
		)
	else:
		filename += f"potential_{potential_strength:.2f}K"

	return filename


