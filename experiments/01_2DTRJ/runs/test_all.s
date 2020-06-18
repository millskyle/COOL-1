#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu_any
#SBATCH --nodelist=moonracer
#SBATCH -t 1-00:00:00
#SBATCH -J placeholder
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G

args="--ent_coef=0.00001 --n_steps=320 --learning_rate=1e-6 --gamma=0.98 --salt=likemoonracer2"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24

source $HOME/.bashrc
conda activate COOL

export COOL_HOME=/home/kmills/COOL

python='python -u'
restore_mode='best'

export LATTICE_L=8
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p79.000_59ee0e6c-60c0-47d5-9987-27797997f8f7_1100000.zip" --codename="shivering-apple" $args

export LATTICE_L=10
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p49.000_f89f913c-0ddf-4393-973f-e14e2ba7c4dd_1650000.zip" --codename="muddled-hat" $args

export LATTICE_L=16
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p42.500_38d6b740-8fbd-48b2-a6dc-8b731c562fab_1650000.zip" --codename="amazing-motor" $args

export LATTICE_L=14
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p35.500_b4fceb25-6245-4173-83ab-e47678ede3f9_1300000.zip" --codename="colorful-dog" $args


export LATTICE_L=6
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p84.000_53dc96d3-bc37-4bc9-8d29-797c59e0778f_550000.zip" --codename="doubtful-series" $args

export LATTICE_L=12
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p37.500_1ca8725f-bc39-440f-940b-2e7d229bfddb_400000.zip" --codename="excellent-taste" $args


export LATTICE_L=4
$python train_val.py --test --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/${LATTICE_L}x$LATTICE_L/validation/ --checkpoint="archived_p90.500_ecd4f6ad-4be5-4f55-81e3-92b30fe4b263_100000.zip" --codename="amusing-reference" $args

