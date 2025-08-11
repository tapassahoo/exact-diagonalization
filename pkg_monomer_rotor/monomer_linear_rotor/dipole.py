import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy import Rational
import warnings

def precompute_monomer_linear_rotor_dipole_terms(JM_list, potential_strength: float):
	"""
	Precompute dipole interaction matrix elements for a linear polar rotor 
	in an external electric field (aligned along z-axis).

	Parameters:
		JM_list (ndarray): Array of shape (N, 2) with rows as (J, M).
		potential_strength (float): Orienting potential strength in cm^-1.

	Returns:
		dict: {(J, Jp, M): value} matrix elements for the Hamiltonian.
	"""
	dipole_terms = {}
	J_values = np.unique(JM_list[:, 0])  # Extract unique J values

	for J in J_values:
		for Jp in J_values:
			min_J = min(J, Jp)
			for M in range(-int(min_J), int(min_J) + 1):
				try:
					# Wigner 3j symbols for the dipole operator (rank 1 spherical tensor)
					w1 = float(wigner_3j(Rational(J), 1, Rational(Jp), 0, 0, 0).evalf())
					w2 = float(wigner_3j(Rational(J), 1, Rational(Jp), -Rational(M), 0, Rational(M)).evalf())

					# Prefactor from angular momentum algebra
					# Note:
					# For integer M, (-1)**(-M) == (-1)**M because:
					#   (-1)**M alternates between +1 (M even) and -1 (M odd).
					#   The inverse power simply flips sign in the exponent,
					#   but since the sequence is periodic with period 2, the value is unchanged.
					# Therefore:
					#   -(-1)**(-M) == -(-1)**M
					#
					# The prefactor expression thus remains identical whether you use (-M) or M in the exponent.
					#prefactor = -(-1)**(-M) * potential_strength * np.sqrt((2 * J + 1) * (2 * Jp + 1)) 
					prefactor = -(-1)**M * potential_strength * np.sqrt((2 * J + 1) * (2 * Jp + 1))

					# Store non-zero matrix elements
					dipole_terms[(J, Jp, M)] = prefactor * w1 * w2

				except Exception as e:
					warnings.warn(f"Wigner 3j symbol computation failed for J={J}, J'={Jp}, M={M}: {e}")
					continue

	return dipole_terms

