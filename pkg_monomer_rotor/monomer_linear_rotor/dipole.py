from sympy.physics.wigner import wigner_3j
from sympy import Rational
import numpy as np

def precompute_monomer_linear_rotor_dipole_terms(JM_list, potential_strength_K):
	dipole_terms = {}
	J_vals = np.unique(JM_list[:, 0])
	for J in J_vals:
		for Jp in J_vals:
			for M in range(-int(min(J, Jp)), int(min(J, Jp)) + 1):
				try:
					w1 = float(wigner_3j(Rational(J), 1, Rational(Jp), 0, 0, 0).evalf())
					w2 = float(wigner_3j(Rational(J), 1, Rational(Jp), -Rational(M), 0, Rational(M)).evalf())
					prefactor = -(-1)**M * potential_strength_K * np.sqrt((2 * J + 1) * (2 * Jp + 1))
					dipole_terms[(J, Jp, M)] = prefactor * w1 * w2
				except:
					continue
	return dipole_terms

