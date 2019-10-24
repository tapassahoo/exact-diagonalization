#!/bin/bash
#SBATCH --job-name=diag1g1
#SBATCH --output=log1g1.out
#SBATCH --time=0-08:00

#SBATCH --mem-per-cpu=1GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python diag_dimer_asym_rot.py 1 1
#python diag_dimer_asym_rot_basis_saved.py 3 8
#python exact_energy_monomer_nonlinear_rotor.py 6 40
