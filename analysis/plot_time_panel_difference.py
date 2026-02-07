import json
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Load both 1500-gen and 500-gen data for completeness
long_file = 'results/experiment_results_1767414768.json'
with open(long_file) as f:
    data = json.load(f)

by_p = {}
for r in data['results']:
    p = r['p']
    by_p.setdefault(p, []).append(r['history'])

# Only plot for p=0.75 and p=0.9
for metric, fname, ylabel in [
    ('hamming_distance', 'results/time_panel_diff_hamming.png', 'Hamming distance'),
    ('genetic_entropy', 'results/time_panel_diff_entropy.png', 'Genetic entropy')
]:
    if 0.75 in by_p and 0.9 in by_p:
        arr_075 = np.array([h[metric] for h in by_p[0.75]])
        arr_09 = np.array([h[metric] for h in by_p[0.9]])
        mean_075 = arr_075.mean(axis=0)
        mean_09 = arr_09.mean(axis=0)
        sem_075 = arr_075.std(axis=0, ddof=1) / sqrt(arr_075.shape[0])
        sem_09 = arr_09.std(axis=0, ddof=1) / sqrt(arr_09.shape[0])
        diff = mean_075 - mean_09
        sem_diff = np.sqrt(sem_075**2 + sem_09**2)
        gens = np.arange(0, len(mean_075)*10, 10)

        plt.figure(figsize=(6,4))
        plt.plot(gens, diff, color='darkorange', label='p=0.75 minus p=0.9')
        plt.fill_between(gens, diff - sem_diff, diff + sem_diff, color='orange', alpha=0.3, label='mean ± SEM')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel(f'Difference in {ylabel}')
        plt.title(f'Difference in {ylabel} (p=0.75 - p=0.9)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"✓ Saved {fname}")
    else:
        print(f"p=0.75 or p=0.9 missing for {metric}")
