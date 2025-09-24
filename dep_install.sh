#!/bin/bash

#SBATCH -A kite_gpu
#SBATCH -t 0:10:0
#SBATCH -p gpu
#SBATCH --gres=gpu:1
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

pip install torch-scatter
pip install seaborn
pip install matplotlib
pip install scikit-learn
