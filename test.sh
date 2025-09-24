#!/bin/bash

#SBATCH -A kite_gpu
#SBATCH -t 0:30:0
#SBATCH --output test_pointnet_stack_40_5fold_4_full.out
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -C gpu32g
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH -c 4

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export WANDB_MODE=offline

module load python3/3.10.9

source milenv/bin/activate

python mm.py test mmr_act pointnet --load pointnet_stack40_5fold_4_full -config ./configs/action/mmr_action_stack_40_point_carelab_5fold_4.toml -w 0 -a gpu
