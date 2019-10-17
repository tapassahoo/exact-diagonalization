#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --output=diag.out
#SBATCH --time=1-00:30

#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
#python exact_energy_dimer_nonlinear_rotors.py 2 6
python exact_energy_monomer_nonlinear_rotor.py 6 40
