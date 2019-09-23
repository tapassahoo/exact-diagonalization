#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --output=diag.out
#SBATCH --time=0-00:30

#SBATCH --mem-per-cpu=4096mb
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python exact_energy_nonlinear_rotors.py 2 6
