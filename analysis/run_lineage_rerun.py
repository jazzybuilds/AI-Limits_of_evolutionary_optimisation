"""Small rerun to collect lineage data for TMRCA and Muller plots.

This script runs a small number of replicates for selected p values
with lineage tracking enabled and saves results to `results_lineage/`.
"""
import sys
from pathlib import Path
# ensure project root is on sys.path so imports work when running from analysis/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import run_full_experiment


if __name__ == '__main__':
    # Target p values to re-run with lineage tracking
    p_levels = [0.75, 0.9]
    n_replicates = 5
    results, outfile = run_full_experiment(
        p_levels=p_levels,
        n_replicates=n_replicates,
        genome_length=50,
        K=2,
        generations=500,
        output_dir='results_lineage',
        lineage_tracking=True
    )
    print('Rerun complete. Results saved to', outfile)
