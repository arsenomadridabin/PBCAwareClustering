#!/bin/bash
#SBATCH -N 60                    # number of nodes
#SBATCH -n 20                   # number of MPI processes
##SBATCH -c 12                  # 6 threads per MPI process
#SBATCH -t 15:00:00
#SBATCH -p checkpt
#SBATCH -A hpc_bb_karki3
#SBATCH -o  gb.out
#SBATCH -e  err.out

source /scratch/ashaky3/.env/bin/activate

python cluster_all_parallel.py
