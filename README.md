# Neutral Evolution Experiment

Research question: **How does varying the proportion of neutral genetic material affect population diversity in a static fitness landscape?**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Experiment

```bash
python run_experiment.py
```

This will:
- Test 7 neutrality levels from 0% to 90%
- Run 20 replicates per level
- Evolve populations for 500 generations each
- Save results to `results/` directory
- Take approximately 10-30 minutes (depending on your machine)

### 3. Analyze Results

```bash
python analyze_results.py results/experiment_results_*.json
```

Or simply:
```bash
python analyze_results.py
```
(will automatically find the most recent results)

This generates:
- Statistical summary in console
- `diversity_vs_neutrality.png` - main results plot
- `time_series.png` - dynamics over time

## Project Structure

```
├── nk_landscape.py          # NK fitness landscape with neutrality
├── diversity_metrics.py     # Population diversity measurements
├── genetic_algorithm.py     # GA implementation
├── run_experiment.py        # Main experimental pipeline
├── analyze_results.py       # Results analysis and visualization
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Implementation Details

### Genetic Algorithm
- **Genome**: Fixed-length binary strings (default 50 bits)
- **Reproduction**: Asexual (mutation-only)
- **Mutation**: Bit-flip with probability 0.01 per gene
- **Selection**: Tournament selection (size 3)
- **Population**: 100 individuals

### Fitness Landscape (NK Model)
- **N**: Genome length (50)
- **K**: Epistatic interactions (2 by default)
- **Neutrality**: Fraction of genes that don't affect fitness
  - Neutral genes still mutate and contribute to diversity
  - Only functional genes contribute to fitness calculation

### Diversity Metrics
1. **Average Pairwise Hamming Distance**: Mean genetic distance between all pairs
2. **Genetic Entropy**: Shannon entropy averaged across all loci
3. **Unique Genotypes**: Number of distinct genotypes in population

## Customization

Edit parameters in `run_experiment.py`:

```python
results, output_file = run_full_experiment(
    neutrality_levels=[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
    n_replicates=20,      # More replicates = better statistics
    genome_length=50,     # Longer genomes = more complexity
    K=2,                  # Higher K = more rugged landscape
    generations=500,      # More generations = longer convergence
    output_dir='results'
)
```

## Expected Results

Based on neutral network theory, we expect:
- **Low neutrality (0-20%)**: Low diversity - strong selection pressure
- **Intermediate neutrality (30-60%)**: **Highest diversity** - balance of selection and drift
- **High neutrality (70-90%)**: Moderate diversity - mostly neutral drift

The "sweet spot" of intermediate neutrality should maximize diversity while maintaining viable fitness.

## Computational Requirements

**Default settings** (7 levels × 20 replicates × 500 generations):
- Time: 10-30 minutes
- Memory: < 500 MB
- Cores: Uses 1 core (can parallelize if needed)

**For thorough analysis** (10 levels × 50 replicates × 1000 generations):
- Time: 1-3 hours
- Consider running overnight

## Troubleshooting

**Import errors**: Ensure requirements are installed
```bash
pip install -r requirements.txt
```

**Slow runtime**: Reduce parameters
```python
n_replicates=10  # instead of 20
generations=300  # instead of 500
```

**No plots showing**: Check if matplotlib backend is set
```bash
export MPLBACKEND=TkAgg  # On macOS/Linux
```

## References

This implementation is inspired by:
- Kimura, M. (1983). *The Neutral Theory of Molecular Evolution*
- Kauffman, S. A. (1993). *The Origins of Order* (NK model)
- Wagner, A. (2008). "Neutralism and selectionism: A network-based reconciliation"

## Output Files

After running experiments, you'll find:

```
results/
├── experiment_results_<timestamp>.json  # Raw data
├── diversity_vs_neutrality.png         # Main results
└── time_series.png                     # Temporal dynamics
```

The JSON file contains complete data for custom analysis if needed.
