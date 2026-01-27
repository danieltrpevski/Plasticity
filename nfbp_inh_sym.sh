#!/bin/bash -l
#SBATCH --job-name="nfbp_inh_sym"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=2-00:00:00
#SBATCH -p long
#SBATCH --mem=440GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128

#SBATCH -e nsyml.e
#SBATCH -o nsyml.o

srun python3 nfbp_inh_sym_mpi.py
