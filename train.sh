#!/bin/bash

#SBATCH -A kite_gpu
#SBATCH -t 0-12:0:0
#SBATCH --output out_pointnet_zoned_carelab_act_test.out
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH -c 4
#SBATCH --mail-user korosh.roohi9731@gmail.com
#SBATCH --mail-type ALL

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export WANDB_MODE=offline

module load python3/3.10.9

source milenv/bin/activate

python mm.py train mmr_act pointnet-film --save-name pointnet-film_stack40_act -config ./configs/action/mmr_action_stack_40_point_carelab_5fold_0.toml -a gpu
