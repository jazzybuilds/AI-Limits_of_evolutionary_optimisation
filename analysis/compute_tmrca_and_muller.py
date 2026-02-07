"""Compute TMRCA from lineage logs and produce Muller plots.

Reads results JSON (with lineage data) and outputs a CSV of TMRCA per replicate
and Muller plot figures per replicate.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results_lineage")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def load_latest_results(path):
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError("No results_lineage JSON files found")
    latest = files[-1]
    with latest.open() as fh:
        data = json.load(fh)
    return data, latest


def build_parent_map(parent_log):
    # parent_log: list of [child_id, parent_id, generation]
    parent = {}
    birth_gen = {}
    for entry in parent_log:
        child, p, gen = entry
        parent[child] = p
        birth_gen[child] = gen
    return parent, birth_gen


def ancestor_list(id_, parent):
    path = []
    cur = id_
    while cur is not None and cur != -1:
        path.append(cur)
        cur = parent.get(cur, -1)
        if cur == -1:
            path.append(-1)
            break
    return path


def founder_of(id_, parent):
    cur = id_
    while True:
        p = parent.get(cur, -1)
        if p == -1 or p is None:
            return cur
        cur = p


def compute_tmrca_for_ids(final_ids, parent, birth_gen):
    # compute ancestor sets with birth generation
    ancestor_sets = []
    ancestor_gen_maps = []
    for fid in final_ids:
        amap = {}
        cur = fid
        while cur is not None and cur != -1:
            gen = birth_gen.get(cur, None)
            amap[cur] = gen
            cur = parent.get(cur, -1)
            if cur == -1:
                amap[-1] = 0
                break
        ancestor_sets.append(set(amap.keys()))
        ancestor_gen_maps.append(amap)

    common = set.intersection(*ancestor_sets)
    if not common:
        return None

    # choose ancestor with maximal birth generation as MRCA
    best = None
    best_gen = -1
    for a in common:
        # get generation from first map that contains it
        g = None
        for amap in ancestor_gen_maps:
            if a in amap:
                g = amap[a]
                break
        if g is None:
            g = 0
        if g > best_gen:
            best_gen = g
            best = a

    return best, best_gen


def plot_muller(snapshots, parent, outpath):
    # snapshots: list of {'generation': gen, 'individuals': [(id, bitstr), ...]}
    # determine founder for every id
    all_ids = {id_ for s in snapshots for id_, _ in s['individuals']}
    founder = {}
    for id_ in all_ids:
        founder[id_] = founder_of(id_, parent)

    gens = [s['generation'] for s in snapshots]
    founders = sorted({founder[id_] for id_ in all_ids if id_ is not None})

    freq = {f: [] for f in founders}
    for s in snapshots:
        ids = [id_ for id_, _ in s['individuals']]
        counts = Counter(founder[id_] for id_ in ids)
        total = max(1, len(ids))
        for f in founders:
            freq[f].append(counts.get(f, 0) / total)

    # stacked area plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = np.vstack([freq[f] for f in founders])
    ax.stackplot(gens, data, labels=[str(f) for f in founders])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fraction of population')
    ax.set_title('Muller plot (founder lineages)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    data, fp = load_latest_results(RESULTS_DIR)
    print('Processing:', fp)
    tmrca_rows = []

    for rec in data.get('results', []):
        if 'lineage' not in rec:
            continue
        p = rec.get('p') or rec.get('neutrality')
        rep = rec.get('replicate')
        parent_log = rec['lineage'].get('parent_log', [])
        snapshots = rec['lineage'].get('snapshots', [])

        parent, birth_gen = build_parent_map(parent_log)

        if not snapshots:
            continue
        final_snapshot = snapshots[-1]
        final_ids = [id_ for id_, _ in final_snapshot['individuals'] if id_ is not None]

        if not final_ids:
            continue

        mrca = compute_tmrca_for_ids(final_ids, parent, birth_gen)
        if mrca is None:
            tmrca_rows.append({'p': p, 'replicate': rep, 'mrca_id': None, 'mrca_gen': None})
        else:
            tmrca_rows.append({'p': p, 'replicate': rep, 'mrca_id': int(mrca[0]), 'mrca_gen': int(mrca[1])})

        # Muller plot per replicate
        outpath = OUT_DIR / f'muller_p{p}_rep{rep}.png'
        try:
            plot_muller(snapshots, parent, outpath)
        except Exception as e:
            print('Failed to plot muller for', p, rep, e)

    # write simple CSV
    import csv
    outcsv = Path('analysis/tmrca_results.csv')
    with outcsv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['p', 'replicate', 'mrca_id', 'mrca_gen'])
        w.writeheader()
        for r in tmrca_rows:
            w.writerow(r)

    print('Wrote TMCRA CSV and Muller plots to', OUT_DIR)


if __name__ == '__main__':
    main()
