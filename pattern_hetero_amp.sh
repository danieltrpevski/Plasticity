#!/bin/bash -l
#SBATCH --job-name="pattern_hetero_amp"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=main,long

#SBATCH -e phea.e
#SBATCH -o phea.o

srun python3 pattern_hetero_amp_mpi.py
