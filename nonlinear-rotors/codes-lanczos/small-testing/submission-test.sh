#!/bin/bash
#SBATCH --job-name=lnzs-omp
#SBATCH --output=lnzs-omp.txt
#SBATCH --time=1-00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=16
time ./run  10.0  8  100 -10.0 0.0
