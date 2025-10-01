#!/usr/bin/env python3
"""
Script to generate and submit SLURM jobs with parameters as command-line arguments.
Supports grid search over multiple parameter values.
Usage: python run_slurm_job.py [options]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from itertools import product

def parse_value(value_str):
    """Parse a parameter value that could be single or comma-separated list."""
    if ',' in value_str:
        return [v.strip() for v in value_str.split(',')]
    return [value_str]

def parse_numeric_list(value_str, value_type):
    """Parse numeric values (int or float) that could be comma-separated."""
    values = parse_value(value_str)
    return [value_type(v) for v in values]

def create_output_paths(args_dict):
    """Create checkpoint folder and output file paths based on parameters."""
    
    # Create a descriptive name based on important parameters
    # Use fold_number for k-fold CV, subject_id for LOSO CV
    if args_dict['cross_validation'].upper() == 'LOSO':
        cv_info = f"subj{args_dict['subject_id']}"
    else:
        cv_info = f"fold{args_dict['fold_number']}"
    
    exp_name = f"{args_dict['model']}_{args_dict['task']}_seed{args_dict['seed']}_stack{args_dict['stacks']}_{args_dict['cross_validation']}_{cv_info}"
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args_dict['checkpoint_base'], exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create output directory for logs
    output_dir = os.path.join(args_dict['output_base'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path
    output_file = os.path.join(output_dir, f"{exp_name}.out")
    
    return checkpoint_dir, output_file, exp_name

def generate_slurm_script(args_dict, checkpoint_dir, output_file):
    """Generate SLURM script with parameters as command-line flags."""
    
    # Build command-line arguments from config parameters
    cmd_args = f"""--dataset_seed {args_dict['seed']} \\
    --dataset_raw_data_path '{args_dict['raw_data_path']}' \\
    --dataset_processed_data '{args_dict['processed_data']}' \\
    --dataset_cross_validation {args_dict['cross_validation']} \\
    --dataset_num_folds {args_dict['num_folds']} \\
    --dataset_fold_number {args_dict['fold_number']} \\
    --dataset_train_split {args_dict['train_split']} \\
    --dataset_val_split {args_dict['val_split']} \\
    --dataset_test_split {args_dict['test_split']} \\
    --dataset_stacks {args_dict['stacks']} \\
    --dataset_zero_padding {args_dict['zero_padding']} \\
    --dataset_max_points {args_dict['max_points']} \\
    --dataset_subject_id {args_dict['subject_id']} \\
    --save-name '{checkpoint_dir}'"""
    
    # Generate SLURM script
    slurm_script = f"""#!/bin/bash

#SBATCH -A {args_dict['account']}
#SBATCH -t {args_dict['time']}
#SBATCH --output {output_file}
#SBATCH -p {args_dict['partition']}
#SBATCH --gres={args_dict['gres']}
#SBATCH --nodes {args_dict['nodes']}
#SBATCH --mem {args_dict['mem']}
#SBATCH -c {args_dict['cpus']}
#SBATCH --mail-user {args_dict['mail_user']}
#SBATCH --mail-type {args_dict['mail_type']}

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export WANDB_MODE={args_dict['wandb_mode']}

module load {args_dict['module']}

source {args_dict['venv']}/bin/activate

python {args_dict['script']} {args_dict['command']} {args_dict['task']} {args_dict['model']} \\
    --save-name {args_dict['save_name']} \\
    {cmd_args} \\
    -a {args_dict['accelerator']} \\
    -lr {args_dict['learning_rate']}
"""
    
    return slurm_script

class MultiValueAction(argparse.Action):
    """Custom action to handle comma-separated values for grid search."""
    def __call__(self, parser, namespace, values, option_string=None):
        if ',' in values:
            setattr(namespace, self.dest, [v.strip() for v in values.split(',')])
        else:
            # Store as single value, not a list
            setattr(namespace, self.dest, values)

class MultiNumericAction(argparse.Action):
    """Custom action to handle comma-separated numeric values."""
    def __init__(self, option_strings, dest, value_type=int, **kwargs):
        self.value_type = value_type
        super().__init__(option_strings, dest, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        if ',' in values:
            parsed = [self.value_type(v.strip()) for v in values.split(',')]
        else:
            # Store as single value, not a list
            parsed = self.value_type(values)
        setattr(namespace, self.dest, parsed)

def main():
    parser = argparse.ArgumentParser(
        description='Generate and submit SLURM job with parameters. Use comma-separated values for grid search.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single job
  python run_slurm_job.py --seed 20 --subject-id 5
  
  # Grid search over seeds
  python run_slurm_job.py --seed 20,42,123
  
  # Grid search over multiple parameters
  python run_slurm_job.py --seed 20,42 --subject-id 1,2,3 --stacks 20,40
  
  # This will create 2*3*2 = 12 jobs
        """
    )
    
    # Config parameters (from your config file) - with grid search support
    parser.add_argument('--seed', type=str, default='20', action=MultiNumericAction, value_type=int,
                        help='Random seed (comma-separated for grid search, e.g., 20,42,123)')
    parser.add_argument('--raw-data-path', type=str, default='data/raw_carelab_zoned',
                        help='Path to raw data')
    parser.add_argument('--processed-data', type=str, 
                        default='/cluster/projects/kite/koorosh/Data/MiliPointCareLab/data/processed_carelab/mmr_action/seed_20_stacks_40_padd_point_task_action.pkl',
                        help='Path to processed data')
    parser.add_argument('--cross-validation', type=str, default='LOSO', action=MultiValueAction,
                        help='Cross validation method (comma-separated for grid search)')
    parser.add_argument('--num-folds', type=str, default='5', action=MultiNumericAction, value_type=int,
                        help='Number of folds (comma-separated for grid search)')
    parser.add_argument('--fold-number', type=str, default='0', action=MultiNumericAction, value_type=int,
                        help='Fold number (comma-separated for grid search)')
    parser.add_argument('--train-split', type=str, default='0.8', action=MultiNumericAction, value_type=float,
                        help='Training split ratio (comma-separated for grid search)')
    parser.add_argument('--val-split', type=str, default='0.1', action=MultiNumericAction, value_type=float,
                        help='Validation split ratio (comma-separated for grid search)')
    parser.add_argument('--test-split', type=str, default='0.1', action=MultiNumericAction, value_type=float,
                        help='Test split ratio (comma-separated for grid search)')
    parser.add_argument('--stacks', type=str, default='40', action=MultiNumericAction, value_type=int,
                        help='Number of stacks (comma-separated for grid search)')
    parser.add_argument('--zero-padding', type=str, default='per_data_point', action=MultiValueAction,
                        help='Zero padding method (comma-separated for grid search)')
    parser.add_argument('--max-points', type=str, default='22', action=MultiNumericAction, value_type=int,
                        help='Maximum number of points (comma-separated for grid search)')
    parser.add_argument('--subject-id', type=str, default='20', action=MultiNumericAction, value_type=int,
                        help='Subject ID (comma-separated for grid search)')
    
    # Directory parameters
    parser.add_argument('--checkpoint-base', type=str, default='./checkpoints',
                        help='Base directory for checkpoints')
    parser.add_argument('--output-base', type=str, default='./outputs',
                        help='Base directory for output logs')
    
    # SLURM parameters
    parser.add_argument('--account', type=str, default='kite_gpu',
                        help='SLURM account')
    parser.add_argument('--time', type=str, default='0-12:0:0',
                        help='Time limit (format: D-HH:MM:SS)')
    parser.add_argument('--output', type=str, default='out_pointnet_zoned_carelab_act_test.out',
                        help='Output file name (used only if --use-custom-output is set)')
    parser.add_argument('--use-custom-output', action='store_true',
                        help='Use custom output filename instead of auto-generated one')
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
    parser.add_argument('--learning-rate', '-lr', type=str, default='1e-3', action=MultiNumericAction, value_type=float,
                        help='Learning rate (comma-separated for grid search)')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                        help='Print scripts without submitting')
    parser.add_argument('--output-script', type=str, default=None,
                        help='Save generated scripts to files with this prefix instead of submitting')
    
    args = parser.parse_args()
    
    # Identify parameters that support grid search and have multiple values
    grid_params = {
        'seed': args.seed if isinstance(args.seed, list) else [args.seed],
        'cross_validation': args.cross_validation if isinstance(args.cross_validation, list) else [args.cross_validation],
        'num_folds': args.num_folds if isinstance(args.num_folds, list) else [args.num_folds],
        'fold_number': args.fold_number if isinstance(args.fold_number, list) else [args.fold_number],
        'train_split': args.train_split if isinstance(args.train_split, list) else [args.train_split],
        'val_split': args.val_split if isinstance(args.val_split, list) else [args.val_split],
        'test_split': args.test_split if isinstance(args.test_split, list) else [args.test_split],
        'stacks': args.stacks if isinstance(args.stacks, list) else [args.stacks],
        'zero_padding': args.zero_padding if isinstance(args.zero_padding, list) else [args.zero_padding],
        'max_points': args.max_points if isinstance(args.max_points, list) else [args.max_points],
        'subject_id': args.subject_id if isinstance(args.subject_id, list) else [args.subject_id],
        'learning_rate': args.learning_rate if isinstance(args.learning_rate, list) else [args.learning_rate]
    }
    
    # Create all combinations
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    combinations = list(product(*param_values))
    
    total_jobs = len(combinations)
    
    # Check if any parameter actually has multiple values
    has_grid_search = any(len(values) > 1 for values in grid_params.values())
    
    if has_grid_search:
        print(f"Grid search enabled: {total_jobs} job(s) will be created")
        print(f"Parameters with multiple values:")
        for name, values in grid_params.items():
            if len(values) > 1:
                print(f"  - {name}: {values}")
        print()
        
        response = input(f"Do you want to proceed with {total_jobs} jobs? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
        print()
    else:
        print(f"Single job will be created")
        print()
    
    submitted_jobs = []
    
    for idx, combo in enumerate(combinations, 1):
        # Create args dictionary for this combination
        args_dict = {
            'seed': combo[param_names.index('seed')],
            'raw_data_path': args.raw_data_path,
            'processed_data': args.processed_data,
            'cross_validation': combo[param_names.index('cross_validation')],
            'num_folds': combo[param_names.index('num_folds')],
            'fold_number': combo[param_names.index('fold_number')],
            'train_split': combo[param_names.index('train_split')],
            'val_split': combo[param_names.index('val_split')],
            'test_split': combo[param_names.index('test_split')],
            'stacks': combo[param_names.index('stacks')],
            'zero_padding': combo[param_names.index('zero_padding')],
            'max_points': combo[param_names.index('max_points')],
            'subject_id': combo[param_names.index('subject_id')],
            'checkpoint_base': args.checkpoint_base,
            'output_base': args.output_base,
            'account': args.account,
            'time': args.time,
            'partition': args.partition,
            'gres': args.gres,
            'nodes': args.nodes,
            'mem': args.mem,
            'cpus': args.cpus,
            'mail_user': args.mail_user,
            'mail_type': args.mail_type,
            'module': args.module,
            'venv': args.venv,
            'wandb_mode': args.wandb_mode,
            'script': args.script,
            'command': args.command,
            'task': args.task,
            'model': args.model,
            'save_name': args.save_name,
            'accelerator': args.accelerator,
            'learning_rate': combo[param_names.index('learning_rate')]
        }
        
        # Create output paths based on important parameters
        checkpoint_dir, output_file, exp_name = create_output_paths(args_dict)
        
        # Use custom output if specified
        if args.use_custom_output:
            output_file = args.output
        
        if has_grid_search:
            print(f"[{idx}/{total_jobs}] Experiment: {exp_name}")
        else:
            print(f"Experiment name: {exp_name}")
        print(f"  Checkpoint directory: {checkpoint_dir}")
        print(f"  Output file: {output_file}")
        
        # Generate SLURM script
        slurm_script = generate_slurm_script(args_dict, checkpoint_dir, output_file)
        
        if args.dry_run:
            print("  Generated SLURM script:")
            print("  " + "=" * 78)
            for line in slurm_script.split('\n'):
                print("  " + line)
            print("  " + "=" * 78)
            print()
            continue
        
        if args.output_script:
            script_file = f"{args.output_script}_{idx}.sh" if has_grid_search else f"{args.output_script}.sh"
            with open(script_file, 'w') as f:
                f.write(slurm_script)
            os.chmod(script_file, 0o755)
            print(f"  SLURM script saved to: {script_file}")
            print()
            continue
        
        # Submit job using sbatch
        try:
            result = subprocess.run(
                ['sbatch'],
                input=slurm_script.encode(),
                capture_output=True,
                text=False
            )
            
            if result.returncode == 0:
                job_info = result.stdout.decode().strip()
                print(f"  {job_info}")
                submitted_jobs.append((exp_name, job_info))
            else:
                print(f"  Error submitting job:")
                print(f"  {result.stderr.decode()}")
                
        except FileNotFoundError:
            print("  Error: sbatch command not found. Are you on a SLURM cluster?")
            sys.exit(1)
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)
        
        print()
    
    if submitted_jobs and not args.dry_run and not args.output_script:
        print(f"\nSummary: {len(submitted_jobs)} job(s) submitted successfully!")
        for exp_name, job_info in submitted_jobs:
            print(f"  - {exp_name}: {job_info}")

if __name__ == '__main__':
    main()
