#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p cpu_g3
#SBATCH -t 1-00:00:00
#SBATCH -J placeholder
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G

export OMP_NUM_THREADS=8
export OMP_NUM_THREADS=20 #moonracer


source $HOME/.bashrc
conda activate COOL
export LATTICE_L=$1

export COOL_HOME=/mount/shockwave/kmills/1q/COOL
python train_val.py --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --ent_coef=0.00001 --n_steps=320 --learning_rate=1e-6 --gamma=0.98 --salt=likemoonracer2 

