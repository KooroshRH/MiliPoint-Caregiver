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

def create_processed_data_path(args_dict):
    """Create dynamic processed data path based on seed, stacks, sampling_rate, max_points, and zero_padding."""
    base_path = args_dict['processed_data_base']
    seed = args_dict['seed']
    stacks = args_dict['stacks']
    sampling_rate = args_dict['sampling_rate']
    max_points = args_dict['max_points']
    zero_padding = args_dict['zero_padding']
    task = args_dict['processed_data_task']

    # Create filename: seed_20_stacks_40_srate_1_maxpts_22_padd_point_task_action.pkl
    filename = f"seed_{seed}_stacks_{stacks}_srate_{sampling_rate}_maxpts_{max_points}_padd_{zero_padding}_task_{task}.pkl"
    processed_data_path = os.path.join(base_path, filename)

    return processed_data_path

def create_output_paths(args_dict):
    """Create checkpoint folder and output file paths based on parameters."""

    # Create a descriptive name based on important parameters
    # Use fold_number for k-fold CV, subject_id for LOSO CV
    if args_dict['cross_validation'].upper() == 'LOSO':
        cv_info = f"subj{args_dict['subject_id']}"
    else:
        cv_info = f"fold{args_dict['fold_number']}"

    # Build ablation study postfix based on model hyperparameters
    ablation_postfix = ""
    if args_dict['model'] == 'dgcnn_aux_fusion_t':
        ablation_parts = []

        # Check for temporal layers = 0 (w/o Temporal Transformer)
        if args_dict.get('model_temporal_layers') == 0:
            ablation_parts.append("woTemporal")
        elif args_dict.get('model_temporal_layers') is not None and args_dict['model_temporal_layers'] != 1:
            ablation_parts.append(f"T{args_dict['model_temporal_layers']}")

        # Check for no FiLM modulation (w/o Auxiliary Modulation)
        if args_dict.get('model_no_film_modulation', False):
            ablation_parts.append("woFiLM")

        # Check for no temporal pos embed (w/o Learnable Pos. Embed)
        if args_dict.get('model_no_temporal_pos_embed', False):
            ablation_parts.append("woPosEmbed")

        # Check for aux_dim = 0 (w/o Auxiliary Features)
        if args_dict.get('model_aux_dim') == 0:
            ablation_parts.append("woAux")
        elif args_dict.get('model_aux_dim') is not None and args_dict['model_aux_dim'] != 4:
            ablation_parts.append(f"aux{args_dict['model_aux_dim']}")

        # Check for non-default k value
        if args_dict.get('model_k') is not None and args_dict['model_k'] != 30:
            ablation_parts.append(f"k{args_dict['model_k']}")

        # Check for non-default temporal heads
        if args_dict.get('model_temporal_heads') is not None and args_dict['model_temporal_heads'] != 4:
            ablation_parts.append(f"H{args_dict['model_temporal_heads']}")

        if ablation_parts:
            ablation_postfix = "_" + "_".join(ablation_parts)

    # Build comprehensive experiment name with all important parameters
    exp_name = (f"{args_dict['model']}_{args_dict['task']}_"
                f"seed{args_dict['seed']}_stack{args_dict['stacks']}_"
                f"{args_dict['cross_validation']}_{cv_info}_"
                f"opt{args_dict['optimizer']}_lr{args_dict['learning_rate']}_"
                f"bs{args_dict['batch_size']}_ep{args_dict['max_epochs']}_"
                f"wd{args_dict['weight_decay']}{ablation_postfix}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args_dict['checkpoint_base'], exp_name)
    # os.makedirs(checkpoint_dir, exist_ok=True)

    # Create output directory for logs
    output_dir = os.path.join(args_dict['output_base'], exp_name)
    # os.makedirs(output_dir, exist_ok=True)

    # Add prefix for test mode output files
    mode_prefix = "test_" if args_dict['mode'] == 'test' else ""
    output_file = os.path.join(output_dir, f"{mode_prefix}{exp_name}.out")

    return checkpoint_dir, output_file, exp_name

def generate_slurm_script(args_dict, checkpoint_dir, output_file):
    """Generate SLURM script with parameters as command-line flags."""

    # Build command-line arguments from config parameters
    temporal_flag = "--dataset_use_temporal_format" if args_dict.get('use_temporal_format', False) else ""

    # Build model-specific arguments
    model_args = ""
    if args_dict.get('model_temporal_layers') is not None:
        model_args += f" \\\n    --model_temporal_layers {args_dict['model_temporal_layers']}"
    if args_dict.get('model_temporal_heads') is not None:
        model_args += f" \\\n    --model_temporal_heads {args_dict['model_temporal_heads']}"
    if args_dict.get('model_aux_dim') is not None:
        model_args += f" \\\n    --model_aux_dim {args_dict['model_aux_dim']}"
    if args_dict.get('model_k') is not None:
        model_args += f" \\\n    --model_k {args_dict['model_k']}"
    if args_dict.get('model_no_film_modulation', False):
        model_args += " \\\n    --model_no_film_modulation"
    if args_dict.get('model_no_temporal_pos_embed', False):
        model_args += " \\\n    --model_no_temporal_pos_embed"

    # Build test/explainability flags (only for test mode)
    test_flags = ""
    if args_dict['mode'] == 'test':
        if args_dict.get('visualize', False):
            test_flags += " \\\n    --visualize"
        if args_dict.get('explainability', False):
            test_flags += " \\\n    --explainability"
            if args_dict.get('explainability_samples', 5) != 5:
                test_flags += f" \\\n    --explainability_samples {args_dict['explainability_samples']}"

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
    --dataset_sampling_rate {args_dict['sampling_rate']} \\
    --dataset_zero_padding {args_dict['zero_padding']} \\
    --dataset_max_points {args_dict['max_points']} \\
    --dataset_subject_id {args_dict['subject_id']} \\
    {temporal_flag}{model_args}{test_flags}"""
    
    # Determine if we're in train or test mode
    mode = args_dict['mode']
    if mode == 'train':
        model_path_arg = f"--save-name '{checkpoint_dir}'"
    else:  # test mode
        model_path_arg = f"--load '{checkpoint_dir}'"

    # Check if model requires 32GB GPU memory
    model_name = args_dict['model'].lower()
    gpu_constraint = ""

    # If user specified a GPU constraint, use it
    if args_dict.get('gpu_constraint'):
        gpu_constraint = f"\n#SBATCH -C {args_dict['gpu_constraint']}"
    else:
        # Otherwise, auto-detect based on model name
        # List of models that need 32GB GPU
        models_requiring_32gb = ['deepgcn', 'pointmlp']
        if any(model in model_name for model in models_requiring_32gb):
            gpu_constraint = "\n#SBATCH -C gpu32g"

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
#SBATCH --mail-type {args_dict['mail_type']}{gpu_constraint}

TMP=~/tmp
TMPDIR=~/tmp
TEMP=~/tmp
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export WANDB_MODE={args_dict['wandb_mode']}

module load {args_dict['module']}

source {args_dict['venv']}/bin/activate

python {args_dict['script']} {mode} {args_dict['task']} {args_dict['model']} \\
    {model_path_arg} \\
    {cmd_args} \\
    -a {args_dict['accelerator']} \\
    -opt {args_dict['optimizer']} \\
    -lr {args_dict['learning_rate']} \\
    -m {args_dict['max_epochs']} \\
    -b {args_dict['batch_size']} \\
    -wd {args_dict['weight_decay']}
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
  # Single training job
  python run_slurm_job.py --seed 20 --subject-id 5
  
  # Single test job
  python run_slurm_job.py --mode test --seed 20 --subject-id 5
  
  # Grid search over seeds (training)
  python run_slurm_job.py --seed 20,42,123
  
  # Grid search over multiple parameters
  python run_slurm_job.py --seed 20,42 --subject-id 1,2,3 --stacks 20,40
  
  # Grid search over training parameters
  python run_slurm_job.py --learning-rate 1e-3,1e-4 --batch-size 64,128 --optimizer adam,adamw
  
  # This will create 2*3*2 = 12 jobs
        """
    )
    
    # Config parameters (from your config file) - with grid search support
    parser.add_argument('--seed', type=str, default='20', action=MultiNumericAction, value_type=int,
                        help='Random seed (comma-separated for grid search, e.g., 20,42,123)')
    parser.add_argument('--raw-data-path', type=str, default='data/raw_carelab_zoned',
                        help='Path to raw data')
    parser.add_argument('--processed-data-base', type=str, 
                        default='/cluster/projects/kite/koorosh/Data/MiliPointCareLab/data/processed_carelab/mmr_action',
                        help='Base path for processed data (will be combined with seed, stacks, max_points, zero_padding)')
    parser.add_argument('--processed-data-task', type=str, default='action',
                        help='Task name for processed data filename')
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
    parser.add_argument('--sampling-rate', type=str, default='1', action=MultiNumericAction, value_type=int,
                        help='Frame sampling rate (comma-separated for grid search, 1=consecutive, 2=every other frame, etc.)')
    parser.add_argument('--zero-padding', type=str, default='per_data_point', action=MultiValueAction,
                        help='Zero padding method (comma-separated for grid search)')
    parser.add_argument('--max-points', type=str, default='22', action=MultiNumericAction, value_type=int,
                        help='Maximum number of points (comma-separated for grid search)')
    parser.add_argument('--subject-id', type=str, default='20', action=MultiNumericAction, value_type=int,
                        help='Subject ID (comma-separated for grid search)')
    parser.add_argument('--use-temporal-format', action='store_true',
                        help='Use temporal format (T, N, C) instead of concatenated format (T*N, C) - required for temporal models like DGCNNAuxFusionT')

    # Training parameters - with grid search support
    parser.add_argument('-opt', '--optimizer', type=str, default='adam', action=MultiValueAction,
                        help='Pick an optimizer (comma-separated for grid search, e.g., adam,adamw,sgd)')
    parser.add_argument('-lr', '--learning-rate', type=str, default='1e-5', action=MultiNumericAction, value_type=float,
                        help='Initial learning rate (comma-separated for grid search, e.g., 1e-3,1e-4,1e-5)')
    parser.add_argument('-m', '--max-epochs', type=str, default='100', action=MultiNumericAction, value_type=int,
                        help='Maximum number of epochs for training (comma-separated for grid search)')
    parser.add_argument('-b', '--batch-size', type=str, default='128', action=MultiNumericAction, value_type=int,
                        help='Batch size for training and evaluation (comma-separated for grid search)')
    parser.add_argument('-wd', '--weight-decay', type=str, default='1e-5', action=MultiNumericAction, value_type=float,
                        help='Weight decay for optimizer regularization (comma-separated for grid search)')

    # Model hyperparameters for ablation studies
    parser.add_argument('--model-temporal-layers', type=str, default=None, action=MultiNumericAction, value_type=int,
                        help='Number of temporal transformer layers (set to 0 for w/o Temporal Transformer ablation)')
    parser.add_argument('--model-temporal-heads', type=str, default=None, action=MultiNumericAction, value_type=int,
                        help='Number of attention heads in temporal transformer (comma-separated for grid search)')
    parser.add_argument('--model-aux-dim', type=str, default=None, action=MultiNumericAction, value_type=int,
                        help='Number of auxiliary channels (set to 0 for w/o Auxiliary Features ablation)')
    parser.add_argument('--model-k', type=str, default=None, action=MultiNumericAction, value_type=int,
                        help='k-NN parameter for graph construction (comma-separated for grid search)')
    parser.add_argument('--model-no-film-modulation', action='store_true',
                        help='Disable FiLM modulation (w/o Auxiliary Modulation ablation)')
    parser.add_argument('--model-no-temporal-pos-embed', action='store_true',
                        help='Disable temporal positional embeddings (w/o Learnable Pos. Embed ablation)')

    # Directory parameters
    parser.add_argument('--checkpoint-base', type=str, default='/cluster/projects/kite/koorosh/Output/MiliPointCareLab/checkpoints',
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
    parser.add_argument('--mem', type=str, default='64G',
                        help='Memory allocation')
    parser.add_argument('--cpus', type=int, default=4,
                        help='Number of CPUs')
    parser.add_argument('--mail-user', type=str, default='korosh.roohi9731@gmail.com',
                        help='Email for notifications')
    parser.add_argument('--mail-type', type=str, default='ALL',
                        help='Email notification types')
    parser.add_argument('--gpu-constraint', type=str, default=None,
                        help='GPU constraint (e.g., gpu32g for 32GB GPU). Auto-set for deepgcn/pointmlp if not specified')

    # Test/Explainability parameters
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize test result as mp4 (test mode only)')
    parser.add_argument('--explainability', '-explain', action='store_true',
                        help='Run explainability analysis after testing (DGCNN-AFTNet only)')
    parser.add_argument('--explainability-samples', '-explain_samples', type=int, default=5,
                        help='Number of samples to visualize per category (TP/FN) for explainability')

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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--task', type=str, default='mmr_act',
                        help='Task name')
    parser.add_argument('--model', type=str, default='pointnet-film',
                        help='Model name')
    parser.add_argument('--save-name', type=str, default='pointnet-film_stack40_act',
                        help='Save name for the model (deprecated, use --mode instead)')
    parser.add_argument('--accelerator', '-a', type=str, default='gpu',
                        help='Accelerator type')
    
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
        'sampling_rate': args.sampling_rate if isinstance(args.sampling_rate, list) else [args.sampling_rate],
        'zero_padding': args.zero_padding if isinstance(args.zero_padding, list) else [args.zero_padding],
        'max_points': args.max_points if isinstance(args.max_points, list) else [args.max_points],
        'subject_id': args.subject_id if isinstance(args.subject_id, list) else [args.subject_id],
        'optimizer': args.optimizer if isinstance(args.optimizer, list) else [args.optimizer],
        'learning_rate': args.learning_rate if isinstance(args.learning_rate, list) else [args.learning_rate],
        'max_epochs': args.max_epochs if isinstance(args.max_epochs, list) else [args.max_epochs],
        'batch_size': args.batch_size if isinstance(args.batch_size, list) else [args.batch_size],
        'weight_decay': args.weight_decay if isinstance(args.weight_decay, list) else [args.weight_decay],
        # Model hyperparameters
        'model_temporal_layers': [args.model_temporal_layers] if args.model_temporal_layers is not None else [None] if not isinstance(args.model_temporal_layers, list) else args.model_temporal_layers,
        'model_temporal_heads': [args.model_temporal_heads] if args.model_temporal_heads is not None else [None] if not isinstance(args.model_temporal_heads, list) else args.model_temporal_heads,
        'model_aux_dim': [args.model_aux_dim] if args.model_aux_dim is not None else [None] if not isinstance(args.model_aux_dim, list) else args.model_aux_dim,
        'model_k': [args.model_k] if args.model_k is not None else [None] if not isinstance(args.model_k, list) else args.model_k,
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
        print(f"Mode: {args.mode}")
        print(f"Parameters with multiple values:")
        for name, values in grid_params.items():
            if len(values) > 1:
                print(f"  - {name}: {values}")
        print()
        
        response = input(f"Do you want to proceed with {total_jobs} {args.mode} jobs? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
        print()
    else:
        print(f"Single {args.mode} job will be created")
        print()
    
    submitted_jobs = []
    
    for idx, combo in enumerate(combinations, 1):
        # Create args dictionary for this combination
        args_dict = {
            'seed': combo[param_names.index('seed')],
            'raw_data_path': args.raw_data_path,
            'processed_data_base': args.processed_data_base,
            'processed_data_task': args.processed_data_task,
            'cross_validation': combo[param_names.index('cross_validation')],
            'num_folds': combo[param_names.index('num_folds')],
            'fold_number': combo[param_names.index('fold_number')],
            'train_split': combo[param_names.index('train_split')],
            'val_split': combo[param_names.index('val_split')],
            'test_split': combo[param_names.index('test_split')],
            'stacks': combo[param_names.index('stacks')],
            'sampling_rate': combo[param_names.index('sampling_rate')],
            'zero_padding': combo[param_names.index('zero_padding')],
            'max_points': combo[param_names.index('max_points')],
            'subject_id': combo[param_names.index('subject_id')],
            'use_temporal_format': args.use_temporal_format,
            'optimizer': combo[param_names.index('optimizer')],
            'learning_rate': combo[param_names.index('learning_rate')],
            'max_epochs': combo[param_names.index('max_epochs')],
            'batch_size': combo[param_names.index('batch_size')],
            'weight_decay': combo[param_names.index('weight_decay')],
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
            'mode': args.mode,
            'task': args.task,
            'model': args.model,
            'save_name': args.save_name,
            'accelerator': args.accelerator,
            # Model hyperparameters
            'model_temporal_layers': combo[param_names.index('model_temporal_layers')],
            'model_temporal_heads': combo[param_names.index('model_temporal_heads')],
            'model_aux_dim': combo[param_names.index('model_aux_dim')],
            'model_k': combo[param_names.index('model_k')],
            'model_no_film_modulation': args.model_no_film_modulation,
            'model_no_temporal_pos_embed': args.model_no_temporal_pos_embed,
        }
        
        # Create dynamic processed data path
        processed_data_path = create_processed_data_path(args_dict)
        args_dict['processed_data'] = processed_data_path
        
        # Create output paths based on important parameters
        checkpoint_dir, output_file, exp_name = create_output_paths(args_dict)
        
        # Use custom output if specified
        if args.use_custom_output:
            output_file = args.output
        
        if has_grid_search:
            print(f"[{idx}/{total_jobs}] {args.mode.upper()} - Experiment: {exp_name}")
        else:
            print(f"{args.mode.upper()} - Experiment name: {exp_name}")
        print(f"  Processed data: {processed_data_path}")
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
        print(f"\nSummary: {len(submitted_jobs)} {args.mode} job(s) submitted successfully!")
        for exp_name, job_info in submitted_jobs:
            print(f"  - {exp_name}: {job_info}")

if __name__ == '__main__':
    main()