# monomer_linear_rotor

**monomer_linear_rotor** is a Python package for constructing and diagonalizing the quantum Hamiltonian of a linear rotor monomer (e.g., HF, HCl, CO) subjected to an external electric field. The module supports nuclear spin isomer selection (`spinless`, `para`, `ortho`) and employs Wigner 3j formalism to construct the field-dressed Hamiltonian in an angular momentum basis. It is designed for use in quantum molecular dynamics, spectroscopy, and statistical thermodynamics.

---

## 🔍 Key Features

- Generates rotational basis states \((J, M)\) for monomeric linear rotors.
- Supports spin isomer types: `spinless`, `para`, and `ortho`.
- Efficiently constructs sparse Hamiltonian matrices using selection rules.
- Precomputes dipole matrix elements via Wigner 3j coefficients.
- Compatible with SciPy’s sparse solvers (`eigsh`) for efficient diagonalization.
- Ready for extension to thermodynamic or field-dependent observables.

---

## 📦 Installation

Clone the repository and install the package in **editable mode** for development:

```bash
cd /Users/tapas/academic-project/exact-diagonalization/pkg_monomer_rotor
pip install -e .

python -c "import monomer_linear_rotor; print('Module loaded successfully')"


pkg_monomer_rotor/
├── monomer_linear_rotor/
│   ├── __init__.py
│   ├── basis.py          ← quantum number generators
│   ├── dipole.py         ← dipole element precomputation
│   ├── hamiltonian.py    ← build, plot, diagonalize Hamiltonian
│   ├── utils.py          ← hermiticity checker, utility functions
├── main.py               ← script to run full program
├── pyproject.toml
├── README.md
├── LICENSE
├── tests/
│   └── test_monomer_linear_rotor.py

