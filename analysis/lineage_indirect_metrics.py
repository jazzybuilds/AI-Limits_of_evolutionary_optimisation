#!/usr/bin/env python3
"""Compute indirect lineage-collapse metrics from existing results JSONs.

Outputs a CSV and prints grouped summaries by neutrality p.
"""
import json
from pathlib import Path
from collections import defaultdict
import csv
import math

RESULTS_DIR = Path("results")
OUT_CSV = Path("analysis/lineage_indirect_metrics.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_records(path):
    records = []
    for fp in path.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            recs = data["results"]
        else:
            recs = [data]

        for rec in recs:
            # extract p
            p = rec.get("neutrality") if isinstance(rec, dict) else None
            if p is None and isinstance(rec, dict) and "parameters" in rec:
                p = rec["parameters"].get("p")
            if p is None:
                continue

            # find history dict-of-arrays
            history = rec.get("history", {}) if isinstance(rec, dict) else {}
            if isinstance(history, dict) and "generation" in history and "unique_genotypes" in history:
                gens = [int(g) for g in history["generation"]]
                ugs = [float(u) for u in history["unique_genotypes"]]
            else:
                # try list-of-dicts
                gens = []
                ugs = []
                hist_list = history if isinstance(history, list) else []
                for e in hist_list:
                    if not isinstance(e, dict):
                        continue
                    if "generation" in e and "unique_genotypes" in e:
                        gens.append(int(e["generation"]))
                        ugs.append(float(e["unique_genotypes"]))

            if not gens:
                continue

            replicate = rec.get("replicate") if isinstance(rec, dict) else None
            records.append({"p": float(p), "replicate": replicate, "gens": gens, "ugs": ugs})

    return records


def analyze_record(rec):
    gens = rec["gens"]
    ugs = rec["ugs"]
    max_u = max(ugs)
    final_u = ugs[-1]

    # max adjacent drop
    max_drop = 0.0
    gen_of_max_drop = None
    for i in range(len(ugs) - 1):
        drop = ugs[i] - ugs[i + 1]
        if drop > max_drop:
            max_drop = drop
            gen_of_max_drop = gens[i + 1]

    rel_drop = max_drop / max_u if max_u > 0 else 0.0

    # T_half: first generation where ugs <= max_u/2
    half_thresh = max_u / 2.0
    t_half = None
    for g, u in zip(gens, ugs):
        if u <= half_thresh:
            t_half = g
            break

    collapse_flag = rel_drop >= 0.5

    return {
        "p": rec["p"],
        "replicate": rec["replicate"],
        "max_unique": max_u,
        "final_unique": final_u,
        "max_adj_drop": max_drop,
        "rel_drop": rel_drop,
        "gen_of_max_drop": gen_of_max_drop if gen_of_max_drop is not None else math.nan,
        "t_half": t_half if t_half is not None else math.nan,
        "collapse_flag": int(collapse_flag),
    }


def main():
    recs = load_records(RESULTS_DIR)
    if not recs:
        print("No records found")
        return

    rows = []
    for r in recs:
        rows.append(analyze_record(r))

    # write CSV
    keys = ["p", "replicate", "max_unique", "final_unique", "max_adj_drop", "rel_drop", "gen_of_max_drop", "t_half", "collapse_flag"]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # grouped summary
    by_p = defaultdict(list)
    for row in rows:
        by_p[row["p"]].append(row)

    print("Summary by p:")
    for p in sorted(by_p.keys()):
        group = by_p[p]
        n = len(group)
        mean_final = sum(r["final_unique"] for r in group) / n
        median_rel_drop = sorted(r["rel_drop"] for r in group)[len(group)//2]
        prop_collapse = sum(r["collapse_flag"] for r in group) / n
        median_t_half = sorted(r["t_half"] for r in group if not math.isnan(r["t_half"]))
        median_t_half = median_t_half[len(median_t_half)//2] if median_t_half else math.nan
        print(f" p={p}: n={n}, mean_final={mean_final:.2f}, median_rel_drop={median_rel_drop:.2f}, prop_collapse={prop_collapse:.2f}, median_t_half={median_t_half}")

    print(f"Wrote CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
