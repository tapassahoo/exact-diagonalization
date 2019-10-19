#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --output=diag.out
#SBATCH --time=0-08:00

#SBATCH --mem-per-cpu=1GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python diag_dimer_asym_rot.py 4 10
#python exact_energy_monomer_nonlinear_rotor.py 6 40
