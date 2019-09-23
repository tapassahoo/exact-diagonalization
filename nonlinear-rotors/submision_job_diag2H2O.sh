#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --output=diag.out
#SBATCH --time=0-00:30

#SBATCH --mem-per-cpu=4096mb
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
python diag2H2O.py 2 8
