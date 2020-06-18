#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu_any
#SBATCH -t 10-00:00:00
#SBATCH -J placeholder
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

export OMP_NUM_THREADS=8


source $HOME/.bashrc
conda activate COOL
export LATTICE_L=$1

export COOL_HOME=/home/kmills/COOL
python train_val.py --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --salt=likemoonracer2 --gamma=0.99 --n_steps=320 

