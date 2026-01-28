#!/bin/bash -l
#SBATCH --job-name="pattern_homo"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

#SBATCH -e pattern_hoe.e
#SBATCH -o pattern_hoo.o

srun python3 pattern_homo_mpi.py
