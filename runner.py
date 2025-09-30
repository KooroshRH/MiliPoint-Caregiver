#!/usr/bin/env python3
"""
Script to generate and submit SLURM jobs with parameters as command-line arguments.
Usage: python run_slurm_job.py [options]
"""

import argparse
import subprocess
import sys

def generate_slurm_script(args):
    """Generate SLURM script with parameters as command-line flags."""
    
    # Build command-line arguments from config parameters
    cmd_args = f"""--seed {args.seed} \\
    --raw-data-path '{args.raw_data_path}' \\
    --processed-data '{args.processed_data}' \\
    --cross-validation {args.cross_validation} \\
    --num-folds {args.num_folds} \\
    --fold-number {args.fold_number} \\
    --train-split {args.train_split} \\
    --val-split {args.val_split} \\
    --test-split {args.test_split} \\
    --stacks {args.stacks} \\
    --zero-padding {args.zero_padding} \\
    --max-points {args.max_points} \\
    --subject-id {args.subject_id}"""
    
    # Generate SLURM script
    slurm_script = f"""#!/bin/bash

#SBATCH -A {args.account}
#SBATCH -t {args.time}
#SBATCH --output {args.output}
#SBATCH -p {args.partition}
#SBATCH --gres={args.gres}
#SBATCH --nodes {args.nodes}
#SBATCH --mem {args.mem}
#SBATCH -c {args.cpus}
#SBATCH --mail-user {args.mail_user}
#SBATCH --mail-type {args.mail_type}

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export WANDB_MODE={args.wandb_mode}

module load {args.module}

source {args.venv}/bin/activate

python {args.script} {args.command} {args.task} {args.model} \\
    --save-name {args.save_name} \\
    {cmd_args} \\
    -a {args.accelerator} \\
    -lr {args.learning_rate}
"""
    
    return slurm_script

def main():
    parser = argparse.ArgumentParser(description='Generate and submit SLURM job with parameters')
    
    # Config parameters (from your config file)
    parser.add_argument('--seed', type=int, default=20,
                        help='Random seed')
    parser.add_argument('--raw-data-path', type=str, default='data/raw_carelab_zoned',
                        help='Path to raw data')
    parser.add_argument('--processed-data', type=str, 
                        default='/cluster/projects/kite/koorosh/Data/MiliPointCareLab/data/processed_carelab/mmr_action/seed_20_stacks_40_padd_point_task_action.pkl',
                        help='Path to processed data')
    parser.add_argument('--cross-validation', type=str, default='LOSO',
                        help='Cross validation method')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--fold-number', type=int, default=0,
                        help='Fold number')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--stacks', type=int, default=40,
                        help='Number of stacks')
    parser.add_argument('--zero-padding', type=str, default='per_data_point',
                        help='Zero padding method')
    parser.add_argument('--max-points', type=int, default=22,
                        help='Maximum number of points')
    parser.add_argument('--subject-id', type=int, default=20,
                        help='Subject ID')
    
    # SLURM parameters
    parser.add_argument('--account', type=str, default='kite_gpu',
                        help='SLURM account')
    parser.add_argument('--time', type=str, default='0-12:0:0',
                        help='Time limit (format: D-HH:MM:SS)')
    parser.add_argument('--output', type=str, default='out_pointnet_zoned_carelab_act_test.out',
                        help='Output file name')
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition')
    parser.add_argument('--gres', type=str, default='gpu:v100:1',
                        help='GPU resources')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes')
    parser.add_argument('--mem', type=str, default='32G',
                        help='Memory allocation')
    parser.add_argument('--cpus', type=int, default=4,
                        help='Number of CPUs')
    parser.add_argument('--mail-user', type=str, default='korosh.roohi9731@gmail.com',
                        help='Email for notifications')
    parser.add_argument('--mail-type', type=str, default='ALL',
                        help='Email notification types')
    
    # Environment parameters
    parser.add_argument('--module', type=str, default='python3/3.10.9',
                        help='Python module to load')
    parser.add_argument('--venv', type=str, default='milenv',
                        help='Virtual environment path')
    parser.add_argument('--wandb-mode', type=str, default='offline',
                        help='Weights & Biases mode')
    
    # Python script parameters
    parser.add_argument('--script', type=str, default='mm.py',
                        help='Python script to run')
    parser.add_argument('--command', type=str, default='train',
                        help='Command to execute')
    parser.add_argument('--task', type=str, default='mmr_act',
                        help='Task name')
    parser.add_argument('--model', type=str, default='pointnet-film',
                        help='Model name')
    parser.add_argument('--save-name', type=str, default='pointnet-film_stack40_act',
                        help='Save name for the model')
    parser.add_argument('--accelerator', '-a', type=str, default='gpu',
                        help='Accelerator type')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                        help='Print script without submitting')
    parser.add_argument('--output-script', type=str, default=None,
                        help='Save generated script to file instead of submitting')
    
    args = parser.parse_args()
    
    # Generate SLURM script
    slurm_script = generate_slurm_script(args)
    
    if args.dry_run:
        print("Generated SLURM script:")
        print("=" * 80)
        print(slurm_script)
        print("=" * 80)
        return
    
    if args.output_script:
        with open(args.output_script, 'w') as f:
            f.write(slurm_script)
        import os
        os.chmod(args.output_script, 0o755)
        print(f"SLURM script saved to: {args.output_script}")
        print(f"Submit with: sbatch {args.output_script}")
        return
    
    # Submit job using sbatch
    try:
        result = subprocess.run(
            ['sbatch'],
            input=slurm_script.encode(),
            capture_output=True,
            text=False
        )
        
        if result.returncode == 0:
            print("Job submitted successfully!")
            print(result.stdout.decode())
        else:
            print("Error submitting job:")
            print(result.stderr.decode())
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on a SLURM cluster?")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
