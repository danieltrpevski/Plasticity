#!/bin/bash -l
#SBATCH --job-name="pattern_hetero"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=main,long

#SBATCH -e pattern_hee.e
#SBATCH -o pattern_heo.o

srun python3 pattern_hetero_mpi.py
