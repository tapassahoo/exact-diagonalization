#!/bin/bash
#SBATCH --job-name=diag1
#SBATCH --output=diag1.out
#SBATCH --time=1-00:30

#SBATCH --mem-per-cpu=100GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python exact_energy_nonlinear_rotors.py 2 10
