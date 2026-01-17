import numpy as np
from termcolor import colored
from pkg_utils.utils import whoami
from pkg_utils.config import *

def generate_monomer_linear_rotor_quantum_numbers(Jmax, spin_isomer_type):
	if spin_isomer_type not in {"spinless", "para", "ortho"}:
		raise ValueError("Choose from 'spinless', 'para', 'ortho'.")
	if spin_isomer_type == "spinless":
		J_values = range(0, Jmax + 1)
	elif spin_isomer_type == "para":
		J_values = range(0, Jmax + 1, 2)
	else:
		J_values = range(1, Jmax + 1, 2)
	return np.array([[J, M] for J in J_values for M in range(-J, J + 1)])

def count_basis_functions(max_angular_momentum_quantum_number, spin_state):
	"""
	Computes and displays the number of real spherical harmonic basis functions
	for a linear rotor system, categorized by nuclear spin isomer type.

	Parameters
	----------
	max_angular_momentum_quantum_number : int
		Maximum angular momentum quantum number (ℓ_max).
	spin_state : str
		Spin isomer type: "spinless", "para", or "ortho".

	Returns
	-------
	dict
		Dictionary with keys:
		- "JM" : Total number of |J,M> functions
		- "JeM": Number of even-J basis functions
		- "JoM": Number of odd-J basis functions
		- "JM_spin_specific": Number for selected spin state
	"""
	spin_state = spin_state.lower()

	if max_angular_momentum_quantum_number < 0:
		raise ValueError("max_angular_momentum_quantum_number must be non-negative.")

	# Total number of basis functions: sum over (2J + 1) from J = 0 to J_max
	JM = (max_angular_momentum_quantum_number + 1) ** 2

	# Even and odd J contributions
	if max_angular_momentum_quantum_number % 2 == 0:
		JeM = (JM + max_angular_momentum_quantum_number + 1) // 2
		JoM = JM - JeM
	else:
		JoM = (JM + max_angular_momentum_quantum_number + 1) // 2
		JeM = JM - JoM

	# Assign basis count based on isomer type
	if spin_state == "spinless":
		njm = JM
	elif spin_state == "para":
		njm = JeM
	elif spin_state == "ortho":
		njm = JoM
	else:
		raise ValueError("Invalid spin_state. Choose from 'spinless', 'para', or 'ortho'.")

	# Display summary
	print(colored("\nNumber of Basis Functions", HEADER_COLOR, attrs=['bold', 'underline']))
	print(colored("[ ] Total |J,M⟩ basis functions:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(JM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Even J basis functions (JeM):".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(JeM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored("[ ] Odd J basis functions (JoM):".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(JoM).ljust(VALUE_WIDTH), VALUE_COLOR))
	print(colored(f"[ ] Basis functions for {spin_state} isomer:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		  colored(str(njm).ljust(VALUE_WIDTH), VALUE_COLOR))

	return {
		"JM": JM,
		"JeM": JeM,
		"JoM": JoM,
		"JM_spin_specific": njm
	}


