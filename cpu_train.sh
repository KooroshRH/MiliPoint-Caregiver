#!/bin/bash

#SBATCH -A kite_cpu
#SBATCH -t 1:0:0
#SBATCH --output out_attdgcnn_carelab_kp_test.out
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

python mm.py train mmr_kp attdgcnn --save-name attdgcnn_stack30_kp -config ./configs/keypoints/mmr_keypoints_stack_30_carelab.toml -a cpu -m 400 -lr 1e-3 
