# pkg_monomer_rotor

A modular Python package for **quantum mechanical modeling of a linear rigid rotor** interacting with external electric fields. The package is designed for high-precision studies of rotational spectra, Hamiltonian structure, and eigenvalue convergence.

---

## 🚀 Features

* Construction of rotational basis sets for linear rigid rotors
* Efficient computation of dipole matrix elements
* Assembly and diagonalization of Hamiltonian matrices
* Visualization of Hamiltonian sparsity and selection rules
* Eigenvalue convergence analysis with respect to ( J_{\max} )
* Designed for extensibility and high-performance workflows

---

## 📁 Project Structure

```
pkg_monomer_rotor/
├── monomer_linear_rotor/
│   ├── __init__.py
│   ├── basis.py
│   ├── dipole.py
│   ├── hamiltonian.py
│   ├── utils.py
├── main.py
├── submit_rotor_jobs_all_at_once_refined.py
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
├── examples/
├── tests/
│   └── test_monomer_linear_rotor.py
├── deploy.sh
```

### Module Overview

* **`basis.py`**
  Generates quantum numbers and basis states for the rotor.

* **`dipole.py`**
  Precomputes dipole matrix elements and selection rules.

* **`hamiltonian.py`**
  Builds the Hamiltonian, performs diagonalization, and provides visualization utilities.

* **`utils.py`**
  Contains helper functions such as Hermiticity checks and validation tools.

* **`main.py`**
  Entry point for running simulations and analysis.

* **`submit_rotor_jobs_all_at_once_refined.py`**
  Batch job submission script for large-scale parameter sweeps (e.g., varying ( J_{\max} )).

---

## ⚙️ Installation

Clone the repository and install locally:

```bash
git clone <repository-url>
cd pkg_monomer_rotor
pip install .
```

For development mode:

```bash
pip install -e .
```

---

## ▶️ Usage

### Running the main program

```bash
python main.py
```

---

### Example: Eigenvalue convergence

```python
import numpy as np
from monomer_linear_rotor.analysis import plot_eigenvalue_convergence

jmax_list = np.array([20, 30, 40], dtype=int)

file_template = (
    "spinless_HF_jmax_{jmax}_field_100.00kV_per_cm/"
    "data/quantum_data_spinless_HF_jmax_{jmax}_field_100.00kV_per_cm.nc"
)

plot_eigenvalue_convergence(
    jmax_list,
    file_template,
    quantum_data_root_dir="path_to_data",
    num_levels_to_show=5
)
```

---

## 📊 Capabilities

* Sparse Hamiltonian structure visualization
* ( \Delta J = \pm 1 ) selection rule verification
* Field-dependent rotational energy shifts
* Convergence analysis with increasing basis truncation

---

## 🧪 Testing

Run unit tests using:

```bash
pytest tests/
```

---

## 📚 Documentation

Detailed documentation and examples are available in the `docs/` and `examples/` directories.

---

## 🧰 Development Notes

* Written in modular and extensible Python
* Designed for integration with HPC workflows
* Compatible with NetCDF-based data pipelines

---

## 📜 License

This project is distributed under the terms of the LICENSE file.

---

## 👤 Author

Dr. Tapas Sahoo
Department of Chemistry
NIT Raipur, India

---

## 🔬 Research Context

This package supports computational studies in:

* Quantum rotational dynamics
* Molecule–field interactions
* Spectroscopic property calculations
* Basis set convergence in quantum systems

---

## 🤝 Contributions

Contributions, suggestions, and collaborations are welcome.

---

## ⭐ Acknowledgment

If you use this code in your research, please consider citing or acknowledging the repository.

