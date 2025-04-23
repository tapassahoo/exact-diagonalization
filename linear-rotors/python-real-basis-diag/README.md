
# Linear Rotor Eigenvalue Computation

This project computes the eigenvalues and eigenfunctions of a linear molecular rotor using a real spherical harmonics basis. It is designed to handle various spin isomers and rotational potentials.

## Developer

**Dr. Tapas Sahoo**

---

## Description

This Python-based code performs exact diagonalization of the Hamiltonian matrix for a linear rotor system. The real basis functions used are real spherical harmonics, which allow efficient numerical computations.

---

## Usage

### Command Line Execution

Run the following command to execute the script:

```bash
python monomer_rotor_real_basis_diagonalization.py <potential_strength> <max_angular_momentum_quantum_number> <spin_state>
```

### Example

```bash
python monomer_rotor_real_basis_diagonalization.py 10.0 2 spinless
```

---

## Input Parameters

- **`potential_strength`**: Strength of the interaction potential (in Kelvin).
- **`max_angular_momentum_quantum_number`**: Highest value of angular momentum quantum number (integer).
- **`spin_state`**: Type of spin isomer (e.g., `spinless`, `para`, `ortho`).

---

## Output

The script generates a NetCDF file containing the following:

- All quantum numbers
- Quantum numbers for the specified spin state
- Eigenvalues
- Real and imaginary parts of the eigenvectors

---

## Reading Output

Use the provided Python script to read and analyze the output from the generated NetCDF file:

### Script: `read_rotor_output.py`

```python
from read_rotor_output import read_quantum_data

filename = "quantum_data_for_H2_spinless_isomer_max_angular_momentum_quantum_number6_potential_strength100.0K_grids_theta17_phi39.nc"
all_qn, spin_qn, eigvals, eigvecs_real, eigvecs_imag, eigvecs = read_quantum_data(filename)
```

---

## Dependencies

The following Python packages are required to run the code:

- Python 3.x
- `numpy`
- `netCDF4`

To install dependencies, use pip:

```bash
pip install numpy netCDF4
```

---

## License

This project is distributed for academic and research purposes. Please cite appropriately if you use this code or its results in publications.

---

## Contact

For any inquiries, reach out to **Dr. Tapas Sahoo**.

---

## Acknowledgements

This project is based on quantum mechanical modeling of molecular rotors and supports high-precision spectroscopy simulations.
