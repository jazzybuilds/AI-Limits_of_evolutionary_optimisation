#!/usr/bin/env python3
"""Plot TMRCA (time to most recent common ancestor) across p values.

Reads `analysis/tmrca_results.csv` and attempts to infer total generations
from `results_lineage` JSON; falls back to 500.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CSV = Path('analysis/tmrca_results.csv')
RESULTS_DIR = Path('results_lineage')
OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)


def infer_generations(results_dir: Path):
    # find any json and read parameters.generations
    for fp in sorted(results_dir.glob('*.json')):
        import json
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        # data may have 'experiment_info' or 'results' entries
        if isinstance(data, dict) and 'experiment_info' in data:
            gen = data['experiment_info'].get('generations')
            if gen:
                return int(gen)
        # else look inside results
        if isinstance(data, dict) and 'results' in data and data['results']:
            r = data['results'][0]
            params = r.get('parameters', {})
            gen = params.get('generations')
            if gen:
                return int(gen)
    return 500


def read_csv(csvp: Path):
    rows = []
    with csvp.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # skip rows with missing mrca_gen
            if not r.get('mrca_gen'):
                continue
            rows.append({'p': float(r['p']), 'replicate': int(r['replicate']), 'mrca_gen': float(r['mrca_gen'])})
    return rows


def plot_tmrca(rows, total_generations, out_prefix: Path = OUT_DIR / 'tmrca_plot'):
    # compute time-to-MRCA = total_generations - mrca_gen
    data = {}
    for r in rows:
        t = total_generations - r['mrca_gen']
        data.setdefault(r['p'], []).append(t)

    ps = sorted(data.keys())
    values = [data[p] for p in ps]

    fig, ax = plt.subplots(figsize=(6, 4))
    # boxplot with explicit styling (blue box)
    boxprops = dict(facecolor='C0', color='black')
    medianprops = dict(color='black', linewidth=2)
    meanprops = dict(marker='D', markeredgecolor='red', markerfacecolor='red')
    # disable fliers (outlier markers) to avoid empty-circle symbols
    bp = ax.boxplot(values, positions=range(len(ps)), widths=0.6, patch_artist=True,
                    showmeans=True, showfliers=False,
                    boxprops=boxprops, medianprops=medianprops, meanprops=meanprops)
    # do not plot individual replicate points (cleaner for publication)

    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel('Neutrality p')
    ax.set_ylabel('Time to MRCA (generations before final)')
    ax.set_title('TMRCA by neutrality level')
    # add legend explaining box vs points and the median/mean markers
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='C0', edgecolor='k', label='Distribution'),
        Line2D([0], [0], color='black', lw=2, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markeredgecolor='red', markersize=8, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()

    png = out_prefix.with_suffix('.png')
    pdf = out_prefix.with_suffix('.pdf')
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print('Saved:', png, pdf)


def main():
    if not CSV.exists():
        print('CSV not found:', CSV)
        return
    rows = read_csv(CSV)
    gens = infer_generations(RESULTS_DIR)
    plot_tmrca(rows, gens)


if __name__ == '__main__':
    main()
