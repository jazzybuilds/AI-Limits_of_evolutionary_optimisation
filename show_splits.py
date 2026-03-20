import json, glob
from pathlib import Path

files = glob.glob('results_stage1/*.json')
path = max(files, key=lambda f: Path(f).stat().st_mtime)
print(f"Loading: {path}\n")
d = json.load(open(path))

for i, r in enumerate(d['results']):
    hist  = r['history']
    train = hist['best_train_score']
    test  = hist['best_test_score']
    print(f"Replicate {i+1} (seed={r['seed']}, best_fitness={r['best_fitness']:.4f}):")
    for gen in [0, 10, 25, 50, 75, 100]:
        if gen < len(train):
            print(f"  Gen {gen:3d}: train={train[gen]:.3f}  test={test[gen]:.3f}")
    t1  = next((g for g, v in enumerate(train) if v >= 1.0), None)
    te1 = next((g for g, v in enumerate(test)  if v >= 1.0), None)
    print(f"  First gen train>=1.0: gen {t1}")
    print(f"  First gen test >=1.0: gen {te1}")
    print()
