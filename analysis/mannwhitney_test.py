import json, glob, os
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else '.'
files = glob.glob(os.path.join(ROOT, 'results', 'experiment_results_*.json'))

data = defaultdict(list)

# Collect final hamming distances; prefer generation==1500 if present
for f in files:
    with open(f) as fh:
        j = json.load(fh)
    for entry in j.get('results', []):
        p = entry.get('neutrality', entry.get('p', None))
        fs = entry.get('final_stats', {})
        hd = fs.get('hamming_distance')
        gen = fs.get('generation')
        if hd is None:
            continue
        # store tuples (generation, hd) to allow filtering later
        data[p].append((gen if gen is not None else -1, hd))

# Prefer entries with generation==1500 if available for a p; otherwise use all final_stats
final_data = {}
for p, vals in data.items():
    vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)
    # if any gen==1500 present, filter to those
    gen1500 = [hd for g, hd in vals if g == 1500]
    if gen1500:
        final_data[p] = gen1500
    else:
        final_data[p] = [hd for g, hd in vals]

print('Collected summary:')
for p in sorted(final_data.keys()):
    arr = np.array(final_data[p])
    if arr.size:
        print(f'p={p}: n={arr.size}, mean={arr.mean():.4f}, median={np.median(arr):.4f}, std={arr.std():.4f}')

p75 = final_data.get(0.75, [])
p90 = final_data.get(0.9, [])

if len(p75) >= 2 and len(p90) >= 2:
    stat, pval = mannwhitneyu(p75, p90, alternative='two-sided')
    # compute rank-biserial effect size (common nonparametric effect size)
    n1 = len(p75); n2 = len(p90)
    U = stat
    rbc = 1 - (2*U)/(n1*n2)
    print('\nMann-Whitney U test (two-sided) comparing p=0.75 vs p=0.9:')
    print(f'U={U}, p-value={pval:.6f}, rank-biserial={rbc:.4f}')
else:
    print('\nInsufficient data for Mann-Whitney test (need >=2 samples per group).')
    print('Available counts: ', {p: len(v) for p, v in final_data.items()})
