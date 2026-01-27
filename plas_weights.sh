#!/bin/bash -l
#SBATCH --job-name="plas_weights"
#SBATCH -A your_account_here
#SBATCH --mem=512GB
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00

#SBATCH -p shared
#SBATCH -n 27

#SBATCH --cpus-per-task=1

#SBATCH -e plasw_error_file.e
#SBATCH -o plasw_output_file.o

srun python3 plas_weights_mpi.py 

