#!/bin/bash
#SBATCH --job-name=diag3
#SBATCH --output=log3.out
#SBATCH --time=7-00:00

#SBATCH --mem-per-cpu=256GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
#python diag_dimer_asym_rot.py 1 1
python diag_dimer_asym_rot_basis_saved.py 5.0 3 8
#python diag_dimer_asym_rot_basis_saved.py 10.0 2 10
