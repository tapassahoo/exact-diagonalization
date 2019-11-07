#!/bin/bash
#SBATCH --job-name=diag4
#SBATCH --output=log4.out
#SBATCH --time=7-00:00

#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
./run 10.05 6 5000
