#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

module load pytorch/2.5.1-py312-cu122-mpi

source ../.env/bin/activate
jupyter nbconvert --to notebook --execute --inplace upo.ipynb
