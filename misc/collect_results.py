"""
Collect LOSO test results from all test_*.out files in outputs/.

Rebuilds from scratch every run — always a consistent snapshot of current results.

Outputs:
  results/results_summary.csv      — one row per (model, subject), all scalar metrics
  results/confusion_matrices.pkl   — dict[(model, subject_id)] = np.array (4x4)
"""
import os
import re
import pickle
import numpy as np
import csv

OUTPUTS_DIR = './outputs'
RESULTS_DIR = './results'


def parse_folder_name(folder):
    """Extract model name and subject_id from experiment folder name."""
    # Format: {model}_{task}_seed{N}_stack{T}_srate{S}_LOSO_subj{ID}_opt...
    m = re.search(r'_LOSO_subj(\d+)_', folder)
    if not m:
        return None, None
    subject_id = int(m.group(1))

    # Model name is everything before the task name
    task_match = re.search(r'_(mmr_act|mmr_kp|mmr_iden)_', folder)
    if not task_match:
        return None, None
    model = folder[:task_match.start()]

    return model, subject_id


def parse_test_output(filepath):
    """Parse a test_*.out file and return dict of metrics + confusion matrix."""
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    result = {
        'test_acc': None,
        'test_top3_acc': None,
        'test_loss': None,
        'merged_acc': None,
        'f1_before_merge': None,
        'f1_merged': None,
        'confusion_matrix': None,
    }

    # Scalar metrics from Lightning table
    for key, pattern in [
        ('test_acc',      r'test_acc\s+([\d.]+)'),
        ('test_top3_acc', r'test_top3_acc\s+([\d.]+)'),
        ('test_loss',     r'test_loss\s+([\d.]+)'),
    ]:
        m = re.search(pattern, content)
        if m:
            result[key] = float(m.group(1))

    # Merged class metrics
    # Be permissive: some logs vary in capitalization / "F1-score" vs "F1 score",
    # and floats may be printed in scientific notation.
    float_re = r'([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)'
    def _search_float(patterns):
        for pat in patterns:
            m = re.search(pat, content, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None

    result['merged_acc'] = _search_float([
        rf'Test\s+Accuracy\s*\(Merged\s+Classes\)\s*:\s*{float_re}',
        rf'Merged\s+Accuracy\s*:\s*{float_re}',
    ])

    result['f1_before_merge'] = _search_float([
        rf'Test\s+F1\s*(?:-|_|\s*)?(?:score)?\s*\(Before\s+Merging\)\s*:\s*{float_re}',
        rf'Test\s+F1\s*(?:-|_|\s*)?(?:score)?\s*Before\s+Merging\s*:\s*{float_re}',
        rf'F1\s*(?:-|_|\s*)?(?:score)?\s*\(Before\s+Merging\)\s*:\s*{float_re}',
    ])

    result['f1_merged'] = _search_float([
        rf'Test\s+F1\s*(?:-|_|\s*)?(?:score)?\s*\(Merged\s+Classes\)\s*:\s*{float_re}',
        rf'Test\s+F1\s*(?:-|_|\s*)?(?:score)?\s*Merged\s+Classes\s*:\s*{float_re}',
        rf'F1\s*(?:-|_|\s*)?(?:score)?\s*\(Merged\s+Classes\)\s*:\s*{float_re}',
    ])

    # Confusion matrix: lines like [[a b c d]\n [e f g h]\n ...]
    cm_match = re.search(r'Confusion Matrix \(Merged Classes\):\s*\n?(\[\[.+?\]\])', content, re.DOTALL)
    if cm_match:
        rows = re.findall(r'\[([\d\s]+)\]', cm_match.group(1))
        if rows:
            matrix = np.array([[int(x) for x in row.split()] for row in rows])
            result['confusion_matrix'] = matrix

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    confusion_matrices = {}

    # Scan all test output files
    for exp_name in sorted(os.listdir(OUTPUTS_DIR)):
        exp_dir = os.path.join(OUTPUTS_DIR, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        model, subject_id = parse_folder_name(exp_name)
        if model is None:
            continue

        # Find test output file
        test_file = os.path.join(exp_dir, f'test_{exp_name}.out')
        if not os.path.isfile(test_file):
            continue

        metrics = parse_test_output(test_file)

        row = {
            'model': model,
            'subject_id': subject_id,
            'test_acc': metrics['test_acc'],
            'test_top3_acc': metrics['test_top3_acc'],
            'test_loss': metrics['test_loss'],
            'merged_acc': metrics['merged_acc'],
            'f1_before_merge': metrics['f1_before_merge'],
            'f1_merged': metrics['f1_merged'],
        }
        rows.append(row)

        if metrics['confusion_matrix'] is not None:
            confusion_matrices[(model, subject_id)] = metrics['confusion_matrix']

    if not rows:
        print("No test output files found.")
        return

    # Sort by model then subject
    rows.sort(key=lambda r: (r['model'], r['subject_id']))

    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, 'results_summary.csv')
    fieldnames = ['model', 'subject_id', 'test_acc', 'test_top3_acc',
                  'test_loss', 'merged_acc', 'f1_before_merge', 'f1_merged']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Append per-model summary (mean ± std)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        f.write('\n')

        models = sorted(set(r['model'] for r in rows))
        for model in models:
            model_rows = [r for r in rows if r['model'] == model]
            summary = {'model': f'{model} (mean)', 'subject_id': len(model_rows)}
            for metric in ['test_acc', 'test_top3_acc', 'test_loss', 'merged_acc',
                           'f1_before_merge', 'f1_merged']:
                vals = [r[metric] for r in model_rows if r[metric] is not None]
                if vals:
                    summary[metric] = f'{np.mean(vals):.4f} ± {np.std(vals):.4f}'
            writer.writerow(summary)

            summary_std = {'model': f'{model} (std)', 'subject_id': ''}
            writer.writerow(summary_std)

    # Save confusion matrices
    pkl_path = os.path.join(RESULTS_DIR, 'confusion_matrices.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(confusion_matrices, f)

    # Print summary to console
    print(f"\nResults collected from {len(rows)} test files")
    print(f"Confusion matrices: {len(confusion_matrices)} entries")
    print(f"\nCSV: {csv_path}")
    print(f"PKL: {pkl_path}")
    print()

    models = sorted(set(r['model'] for r in rows))
    for model in models:
        model_rows = [r for r in rows if r['model'] == model]
        accs = [r['test_acc'] for r in model_rows if r['test_acc'] is not None]
        merged = [r['merged_acc'] for r in model_rows if r['merged_acc'] is not None]
        f1_before = [r['f1_before_merge'] for r in model_rows if r['f1_before_merge'] is not None]
        f1s = [r['f1_merged'] for r in model_rows if r['f1_merged'] is not None]
        print(f"{model} ({len(model_rows)} subjects):")
        if accs:
            print(f"  test_acc:   {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
                  f"[min={min(accs):.4f}, max={max(accs):.4f}]")
        if merged:
            print(f"  merged_acc: {np.mean(merged):.4f} ± {np.std(merged):.4f}  "
                  f"[min={min(merged):.4f}, max={max(merged):.4f}]")
        if f1_before:
            print(f"  f1_before_merge: {np.mean(f1_before):.4f} ± {np.std(f1_before):.4f}")
        if f1s:
            print(f"  f1_merged:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # List missing subjects
        found = sorted(r['subject_id'] for r in model_rows)
        missing = [s for s in range(1, 21) if s not in found]
        if missing:
            print(f"  MISSING subjects: {missing}")
        print()


if __name__ == '__main__':
    main()
