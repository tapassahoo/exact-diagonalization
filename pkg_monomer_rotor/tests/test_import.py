import numpy as np
from scipy.sparse.linalg import eigsh
from monomer_linear_rotor import (
	generate_monomer_linear_rotor_quantum_numbers,
	precompute_monomer_linear_rotor_dipole_terms,
	build_monomer_linear_rotor_hamiltonian
)

def test_basis_size():
	Jmax = 5
	JM_list = generate_monomer_linear_rotor_quantum_numbers(Jmax, "spinless")
	expected_size = sum(2 * J + 1 for J in range(Jmax + 1))
	assert JM_list.shape == (expected_size, 2), "Incorrect number of (J, M) pairs"

def test_hamiltonian_shape():
	Jmax = 4
	B, mu, E = 10.0, 1.0, 50.0
	JM_list = generate_monomer_linear_rotor_quantum_numbers(Jmax, "para")
	dipole_terms = precompute_monomer_linear_rotor_dipole_terms(JM_list, mu, E)
	H = build_monomer_linear_rotor_hamiltonian(JM_list, B, mu, E, dipole_terms)
	dim = JM_list.shape[0]
	assert H.shape == (dim, dim), "Hamiltonian has incorrect shape"

def test_hamiltonian_is_hermitian():
	Jmax = 4
	B, mu, E = 10.0, 1.0, 100.0
	JM_list = generate_monomer_linear_rotor_quantum_numbers(Jmax, "ortho")
	dipole_terms = precompute_monomer_linear_rotor_dipole_terms(JM_list, mu, E)
	H = build_monomer_linear_rotor_hamiltonian(JM_list, B, mu, E, dipole_terms)
	diff = (H - H.getH()).tocoo()
	max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
	assert max_diff < 1e-12, "Hamiltonian is not Hermitian"

def test_eigenvalues_are_real():
	Jmax = 6
	B, mu, E = 10.0, 1.0, 30.0
	JM_list = generate_monomer_linear_rotor_quantum_numbers(Jmax, "spinless")
	dipole_terms = precompute_monomer_linear_rotor_dipole_terms(JM_list, mu, E)
	H = build_monomer_linear_rotor_hamiltonian(JM_list, B, mu, E, dipole_terms)
	eigvals, _ = eigsh(H, k=5, which='SA')
	assert np.all(np.isreal(eigvals)), "Some eigenvalues are not real"

def test_eigenvalues_are_sorted():
	Jmax = 6
	B, mu, E = 10.0, 1.0, 30.0
	JM_list = generate_monomer_linear_rotor_quantum_numbers(Jmax, "spinless")
	dipole_terms = precompute_monomer_linear_rotor_dipole_terms(JM_list, mu, E)
	H = build_monomer_linear_rotor_hamiltonian(JM_list, B, mu, E, dipole_terms)
	eigvals, _ = eigsh(H, k=5, which='SA')
	sorted_eigvals = np.sort(eigvals)
	assert np.allclose(eigvals, sorted_eigvals), "Eigenvalues are not sorted"

