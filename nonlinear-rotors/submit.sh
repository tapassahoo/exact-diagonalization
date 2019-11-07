#!/bin/bash
#SBATCH --job-name=diag4
#SBATCH --output=log4.out
#SBATCH --time=7-00:00

#SBATCH --mem-per-cpu=812GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
#python diag_dimer_asym_rot.py 1 1
python diag_dimer_asym_rot_basis_saved.py 10.05 4 8
python diag_dimer_asym_rot_basis_saved.py 10.05 4 10
