#!/bin/bash -l
#SBATCH --job-name="nfbp_inh_upto3"
#SBATCH -A your_account_here
#SBATCH --mail-type=ALL
#SBATCH --time=23:59:59
#SBATCH -p main
#SBATCH --mem=440GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128

#SBATCH -e upto3_exc.e
#SBATCH -o upto3_exc.o

srun python3 nfbp_inh_upto3_mpi.py
