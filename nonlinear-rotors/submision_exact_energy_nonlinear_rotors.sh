#!/bin/bash
#SBATCH --job-name=diag3
#SBATCH --output=diag3.out
#SBATCH --time=1-00:30

#SBATCH --mem-per-cpu=180GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python exact_energy_nonlinear_rotors.py 4 6
