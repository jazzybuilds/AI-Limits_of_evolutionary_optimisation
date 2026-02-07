#!/usr/bin/env python3
"""Aggregate `unique_genotypes` over generations by neutrality `p` and plot.

Saves PNG and PDF to `figures/unique_genotypes_over_time.png` and `.pdf`.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path):
    by_p = defaultdict(list)
    for fp in path.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        records = []
        # support two file-level formats: a single replicate dict, or a top-level 'results' list
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            records = data["results"]
        else:
            records = [data]

        for rec in records:
            # try to find p value in the record
            p = None
            if isinstance(rec, dict):
                # common keys: 'neutrality' or nested parameters
                if "neutrality" in rec:
                    p = rec.get("neutrality")
                if p is None and "parameters" in rec and isinstance(rec["parameters"], dict):
                    p = rec["parameters"].get("p", None)
                if p is None and "params" in rec and isinstance(rec["params"], dict):
                    p = rec["params"].get("p", None)
                if p is None:
                    p = rec.get("p", None)

            if p is None:
                # skip this record
                continue

            # find history. Two formats supported:
            # 1) history: list of dicts [{generation:..., unique_genotypes:...}, ...]
            # 2) history: dict of arrays {"generation": [...], "unique_genotypes": [...], ...}
            history = []
            if isinstance(rec, dict):
                history = rec.get("history", []) or []
            if not history and "generations" in rec:
                history = rec["generations"]

            rep = {}
            if isinstance(history, dict):
                # history as dict of arrays
                gens = history.get("generation") or history.get("generations")
                ugs = None
                # possible keys
                for key in ("unique_genotypes", "unique_genotype_count", "unique_genotypes_count"):
                    if key in history:
                        ugs = history[key]
                        break
                if gens and ugs and len(gens) == len(ugs):
                    for gen, ug in zip(gens, ugs):
                        try:
                            rep[int(gen)] = float(ug)
                        except Exception:
                            continue
            else:
                # history as list of dicts
                for entry in history:
                    if not isinstance(entry, dict):
                        continue
                    gen = entry.get("generation")
                    ug = entry.get("unique_genotypes")
                    # fallback: some logs might use 'unique_genotype_count' or similar
                    if ug is None:
                        ug = entry.get("unique_genotype_count")
                    if gen is None or ug is None:
                        continue
                    rep[int(gen)] = float(ug)

            if rep:
                by_p[float(p)].append(rep)

    return by_p


def aggregate(by_p):
    # For each p, get sorted union of generations and compute mean/std across reps
    out = {}
    for p, reps in sorted(by_p.items()):
        gens = set()
        for r in reps:
            gens.update(r.keys())
        gens = sorted(gens)

        means = []
        stds = []
        counts = []
        for g in gens:
            vals = [r[g] for r in reps if g in r]
            if vals:
                arr = np.array(vals, dtype=float)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
                counts.append(len(arr))
            else:
                means.append(math.nan)
                stds.append(math.nan)
                counts.append(0)

        out[p] = {
            "generations": gens,
            "mean": means,
            "std": stds,
            "n": counts,
        }
    return out


def plot_aggregated(agg, out_prefix: Path = OUT_DIR / "unique_genotypes_over_time"):
    try:
        plt.style.use("seaborn")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (p, data) in enumerate(sorted(agg.items())):
        gens = np.array(data["generations"])
        mean = np.array(data["mean"])
        std = np.array(data["std"])

        if gens.size == 0:
            continue

        # only plot points where we have at least one replicate
        mask = np.array(data["n"]) > 0
        gens = gens[mask]
        mean = mean[mask]
        std = std[mask]

        color = colors[i % len(colors)]
        ax.plot(gens, mean, label=f"p={p}", color=color)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Unique genotypes (mean ± SD)")
    ax.set_title("Unique genotypes over time by NKp neutrality parameter")
    ax.legend(title="p value", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle=":")
    fig.tight_layout()

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print(f"Saved: {png}, {pdf}")


def plot_traces(by_p, out_prefix: Path = OUT_DIR / "unique_genotypes_traces"):
    plt.figure(figsize=(8, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (p, reps) in enumerate(sorted(by_p.items())):
        color = colors[i % len(colors)]
        # plot each replicate as a thin semi-transparent line
        for rep in reps:
            gens = sorted(rep.keys())
            vals = [rep[g] for g in gens]
            plt.plot(gens, vals, color=color, alpha=0.25, linewidth=0.8)

        # overlay mean
        # compute mean across available gens
        gens_all = sorted({g for r in reps for g in r.keys()})
        mean_vals = []
        for g in gens_all:
            vals = [r[g] for r in reps if g in r]
            mean_vals.append(float(np.mean(vals)) if vals else np.nan)
        plt.plot(gens_all, mean_vals, color=color, linewidth=2.0, label=f"p={p}")

    plt.xlabel("Generation")
    plt.ylabel("Unique genotypes (per replicate)")
    plt.title("Per-replicate traces of unique genotypes over time")
    plt.legend(title="p value", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    print(f"Saved traces: {png}, {pdf}")


def main():
    by_p = load_results(RESULTS_DIR)
    if not by_p:
        print("No result files found or no 'p' parameter present in JSONs.")
        return
    agg = aggregate(by_p)
    plot_aggregated(agg)
    plot_traces(by_p)


if __name__ == "__main__":
    main()
