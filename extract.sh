#!/bin/bash

#SBATCH -A kite_gpu
#SBATCH -t 0-2:0:0
#SBATCH --output extract_carelab_aux_2.out
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --mem 8G
#SBATCH -c 4

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

module load python3/3.10.9

source milenv/bin/activate

python -u misc/carelab_extractor_v2.py \
    --input /cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data/ \
    --output /cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed
