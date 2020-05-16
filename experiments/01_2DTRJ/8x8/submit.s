#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -J placeholder
#SBATCH -c 18
#SBATCH --mem=8G

export LATTICE_L=8
export OMP_NUM_THREADS=40
export COOL_HOME=/home/kmills/1q/COOL
python -u train_val.py --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --salt=$1

