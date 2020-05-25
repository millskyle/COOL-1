#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p cpu
#SBATCH -t 1-00:00:00
#SBATCH -J placeholder
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

args="--ent_coef=0.00001 --n_steps=320 --learning_rate=1e-6 --gamma=0.98 --salt=likemoonracer2"

source $HOME/.bashrc
conda activate COOL

export COOL_HOME=/mount/shockwave/kmills/1q/COOL

export LATTICE_L=14
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="rich-specialist" $args

export LATTICE_L=6
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="majestic-bother" $args

export LATTICE_L=10
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="stupendous-safe" $args

export LATTICE_L=16
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="doubtful-delay" $args

export LATTICE_L=12
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="tender-effective" $args

export LATTICE_L=8
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="motionless-remote" $args

export LATTICE_L=4
python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint='best' --codename="didactic-pin" $args


