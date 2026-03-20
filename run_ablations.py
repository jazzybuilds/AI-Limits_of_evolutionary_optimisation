"""
Ablation runner: Test 0 (no neutrality) and Test 2 (real-valued encoding)
Both on LunarLander, all other hyperparameters held constant.

Results saved to results_stage1/ with descriptive filenames.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from run_stage1 import CONFIG, run_stage1
import copy, json
from pathlib import Path

BASE = copy.deepcopy(CONFIG)
BASE.update({
    "ENV_TYPE":        "lunarlander_continuous",
    "GENOME_ENCODING": "binary",
    "NEUTRALITY_P":    0.75,          # baseline already run; keep consistent
    "MUTATION_RATE":   0.005,
    "POPULATION_SIZE": 20,
    "GENERATIONS":     50,
    "TOURNAMENT_SIZE": 3,
    "N_REPLICATES":    2,
    "SNAPSHOT_GENS":   [0, 10, 20, 30, 40, 50],
})

ABLATIONS = [
    # Test 0: remove neutrality, everything else identical
    {
        "label":        "test0_no_neutrality",
        "NEUTRALITY_P": 0.0,
        "GENOME_ENCODING": "binary",
        "MUTATION_RATE": 0.005,
    },
    # Test 2: real-valued encoding, no neutrality concept
    {
        "label":        "test2_realvalued",
        "NEUTRALITY_P": 0.0,
        "GENOME_ENCODING": "realvalued",
        "MUTATION_RATE": 0.05,   # Gaussian sigma
    },
]

for abl in ABLATIONS:
    label = abl.pop("label")
    cfg = copy.deepcopy(BASE)
    cfg.update(abl)

    print(f"\n{'='*60}")
    print(f"ABLATION: {label}")
    print(f"  NEUTRALITY_P={cfg['NEUTRALITY_P']}  "
          f"GENOME_ENCODING={cfg['GENOME_ENCODING']}  "
          f"MUTATION_RATE={cfg['MUTATION_RATE']}")
    print(f"{'='*60}")

    results, out_file = run_stage1(cfg)

    # Rename output file to include ablation label
    out_path = Path(out_file)
    labelled = out_path.parent / out_path.name.replace(
        f"stage1_{cfg['ENV_TYPE']}_{cfg['CONTROLLER_TYPE']}",
        f"stage1_{cfg['ENV_TYPE']}_{cfg['CONTROLLER_TYPE']}_{label}"
    )
    out_path.rename(labelled)
    print(f"Saved as: {labelled}")
