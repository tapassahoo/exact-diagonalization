****************************************************************************************************
*                                                                                                  *
*      Computation of Eigenvalues and Eigenfunctions of a Linear Rotor                             *
*                 Using Real Spherical Harmonics                                                   *
*                                                                                                  *
*                                  Developed by Dr. Tapas Sahoo                                    *
*                                                                                                  *
****************************************************************************************************

Description:
------------
This program computes the eigenvalues and eigenfunctions of a linear rotor 
in an external potential using a real spherical harmonics basis.

----------------------------------------------------------------------------------------------------
Command to Run:
---------------
Example:
    python monomer_rotor_real_basis_diagonalization.py 10.0 2 spinless

Here,
- 10.0       → Potential strength (in appropriate units)
- 2          → Maximum angular momentum quantum number
- spinless   → Spin isomer specification

----------------------------------------------------------------------------------------------------
Input Parameters:
-----------------
a) Potential strength: 
       The strength of the external potential applied to the rotor.

b) Maximum angular momentum quantum number:
       Sets the size of the angular basis used for the rotor.

c) Spin isomer specification:
       Indicates the nuclear spin symmetry (e.g., 'spinless').

----------------------------------------------------------------------------------------------------
Output:
-------
The program outputs a NetCDF (.nc) data file containing:
- Eigenvalues (rotational energy levels)
- Eigenfunctions (separate arrays for real and imaginary components)
- Quantum numbers associated with the eigenstates

----------------------------------------------------------------------------------------------------
How to Extract and Analyze Data:
--------------------------------
To read the generated NetCDF output file, you may use the following Python snippet:

```python
from netCDF4 import Dataset
import numpy as np

# Replace with your actual output file name
filename = "quantum_data_for_H2_spinless_isomer_max_angular_momentum_quantum_number6_potential_strength100.0K_grids_theta17_phi39.nc"

# Open and read the file
with Dataset(filename, "r") as ncfile:
    eigenvalues = ncfile.variables["eigenvalues"][:]
    eigenvectors_real = ncfile.variables["eigenvectors_real"][:]
    eigenvectors_imag = ncfile.variables["eigenvectors_imag"][:]
    quantum_numbers = ncfile.variables["all_quantum_numbers"][:]

    # Reconstruct complex eigenvectors
    eigenvectors = eigenvectors_real + 1j * eigenvectors_imag

# Print first few eigenvalues as a check
print("First 5 eigenvalues:", eigenvalues[:5])


# Linear Rotor Eigenvalue Computation

This project computes the eigenvalues and eigenfunctions of a linear molecular rotor using a real spherical harmonics basis. It is designed to handle various spin isomers and rotational potentials.

## Developer
Dr. Tapas Sahoo

---

## Description
This Python-based code performs exact diagonalization of the Hamiltonian matrix for a linear rotor system. The real basis functions used are real spherical harmonics, which allow efficient numerical computations.

---

## Usage

### Command Line Execution
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
- NetCDF file containing:
  - All quantum numbers
  - Quantum numbers for the specified spin state
  - Eigenvalues
  - Real and imaginary parts of the eigenvectors

---

## Reading Output
Use the provided Python script to read and analyze the output NetCDF file:

### Script: `read_rotor_output.py`
```python
from read_rotor_output import read_quantum_data

filename = "quantum_data_for_H2_spinless_isomer_max_angular_momentum_quantum_number6_potential_strength100.0K_grids_theta17_phi39.nc"
all_qn, spin_qn, eigvals, eigvecs_real, eigvecs_imag, eigvecs = read_quantum_data(filename)
```

---

## Dependencies
- Python 3.x
- `numpy`
- `netCDF4`

Install dependencies via pip:
```bash
pip install numpy netCDF4
```

---

## License
This project is distributed for academic and research use. Please cite appropriately if used in publications.

---

## Contact
For any queries, reach out to Dr. Tapas Sahoo

---

## Acknowledgement
This project is based on quantum mechanical modeling of molecular rotors and supports high-precision spectroscopy simulations.


