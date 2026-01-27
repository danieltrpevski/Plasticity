#!/bin/bash -l
#SBATCH --job-name="plas_rates"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

#SBATCH -e plasr_error_file.e
#SBATCH -o plasr_output_file.o

srun python3 plas_rates_mpi.py 

