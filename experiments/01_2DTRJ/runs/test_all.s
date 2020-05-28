#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p cpu_g3
#SBATCH -t 1-00:00:00
#SBATCH -J placeholder
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

args="--ent_coef=0.00001 --n_steps=320 --learning_rate=1e-6 --gamma=0.98 --salt=likemoonracer2"

export OMP_NUM_THREADS=32

source $HOME/.bashrc
conda activate COOL

export COOL_HOME=/home/kmills/COOL

python='python -u'

export LATTICE_L=8
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="messy-award" $args


export LATTICE_L=10
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="muddled-hat" $args

export LATTICE_L=16
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="amazing-motor" $args

export LATTICE_L=14
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="colorful-dog" $args

export LATTICE_L=6
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="muddled-hat" $args

export LATTICE_L=12
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="excellent-taste" $args


export LATTICE_L=4
$python train_val.py --test --hamiltonian_directory=${HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="amusing-reference" $args

sbatch --begin='now+1hour' test_all.s 
