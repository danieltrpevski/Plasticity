#!/bin/bash -l
#SBATCH --job-name="pattern_homo_bcm"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

#SBATCH -e pattern_hobe.e
#SBATCH -o pattern_hobo.o

srun python3 pattern_homo_bcm_mpi.py
