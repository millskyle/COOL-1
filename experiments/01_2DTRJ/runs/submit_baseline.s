#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p cpu
#SBATCH -t 1-00:00:00
#SBATCH -J baseline
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export OMP_NUM_THREADS=1

source $HOME/.bashrc
conda activate COOL
export LATTICE_L=$1
export COOL_HOME=/mount/shockwave/kmills/1q/COOL

python baseline.py --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --dbeta=$2


