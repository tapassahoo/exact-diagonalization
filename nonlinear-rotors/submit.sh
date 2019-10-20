#!/bin/bash
#SBATCH --job-name=diag2
#SBATCH --output=log.out
#SBATCH --time=0-08:00

#SBATCH --mem-per-cpu=480GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python diag_dimer_asym_rot.py 2 10
#python exact_energy_monomer_nonlinear_rotor.py 6 40
