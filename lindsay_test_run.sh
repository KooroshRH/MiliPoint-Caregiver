#!/bin/bash

#SBATCH -A kite_gpu
#SBATCH -t 0:1:0
#SBATCH --output lindsay_test_output.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --mem 8G
#SBATCH -c 1
#SBATCH --mail-user korosh.roohi9731@gmail.com
#SBATCH --mail-type ALL

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export WANDB_MODE=offline

module load python3/3.10.9

python lindsay_test.py
