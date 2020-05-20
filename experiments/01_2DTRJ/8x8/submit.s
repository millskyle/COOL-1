#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p cpu_g3
#SBATCH -t 1-00:00:00
#SBATCH -J placeholder
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G

source $HOME/.bashrc
conda activate COOL
export LATTICE_L=16
export OMP_NUM_THREADS=12
export COOL_HOME=/mount/shockwave/kmills/1q/COOL
python train_val.py --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ $@ 

