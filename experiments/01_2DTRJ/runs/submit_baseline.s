#!/bin/bash
#SBATCH -p cpu
#SBATCH -t 1-00:00:00
#SBATCH -J baseline
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

export OMP_NUM_THREADS=4

source $HOME/.bashrc
conda activate COOL
export LATTICE_L=$1
export COOL_HOME=/home/kmills/COOL

python baseline.py --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --dbeta=$2


