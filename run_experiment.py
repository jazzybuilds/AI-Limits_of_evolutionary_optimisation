"""
Main experiment runner for neutrality study.
Systematically varies neutrality parameter (p) and measures population diversity.
"""
import numpy as np
import json
import time
from pathlib import Path
from nk_landscape import NKLandscape
from genetic_algorithm import GeneticAlgorithm


def run_single_trial(p, genome_length=50, K=2, population_size=100,
                     mutation_rate=0.01, generations=500, seed=None,
                     lineage_tracking=False):
    """
    Run a single evolutionary trial with given neutrality level.
    
    Args:
        p: Neutrality probability (0.0 to 1.0) - probability that a mutation is neutral
        genome_length: Length of binary genome
        K: Epistatic interactions parameter
        population_size: Population size
        mutation_rate: Per-bit mutation probability
        generations: Number of generations to evolve
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    # Create fitness landscape
    landscape = NKLandscape(N=genome_length, K=K, p=p, seed=seed)
    
    # Create and run GA
    ga = GeneticAlgorithm(landscape, population_size=population_size,
                         mutation_rate=mutation_rate, seed=seed,
                         lineage_tracking=lineage_tracking)
    ga.initialize_population()
    
    # Run evolution
    history = ga.run(generations, record_interval=10, verbose=False)
    
    # Get final statistics
    final_stats = ga.get_summary_statistics()
    
    result = {
        'p': p,
        'final_stats': final_stats,
        'history': history,
        'parameters': {
            'genome_length': genome_length,
            'K': K,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'generations': generations,
            'seed': seed
        }
    }
    if lineage_tracking:
        # include parent pointers and snapshots
        result['lineage'] = {
            'parent_log': ga.parent_log,
            'snapshots': ga.snapshot_history
        }
    return result


def run_full_experiment(p_levels=None, n_replicates=10, 
                       genome_length=50, K=2, generations=500,
                       output_dir='results', lineage_tracking=False):
    """
    Run complete experiment across multiple neutrality levels (p) with replicates.
    
    Args:
        p_levels: List of neutrality probabilities to test
        n_replicates: Number of replicate runs per p level
        genome_length: Length of binary genome
        K: Epistatic interactions parameter
        generations: Number of generations per run
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all experimental results
    """
    if p_levels is None:
        # Default: test 7 levels from 0% to 90% neutral
        p_levels = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Starting NKp experiment with {len(p_levels)} neutrality levels")
    print(f"Running {n_replicates} replicates per level")
    print(f"Total runs: {len(p_levels) * n_replicates}")
    print(f"Parameters: N={genome_length}, K={K}, generations={generations}")
    print("-" * 60)
    
    all_results = []
    start_time = time.time()
    
    total_runs = len(p_levels) * n_replicates
    current_run = 0
    
    for p in p_levels:
        print(f"\np = {p:.2f} ({int(p*100)}% neutral mutation probability)")
        
        for replicate in range(n_replicates):
            current_run += 1
            
            # Use different seed for each replicate
            seed = replicate * 1000 + int(p * 100)
            
            # Run trial
            result = run_single_trial(
                p=p,
                genome_length=genome_length,
                K=K,
                generations=generations,
                seed=seed,
                lineage_tracking=lineage_tracking
            )
            
            result['replicate'] = replicate
            all_results.append(result)
            
            # Progress update
            if (replicate + 1) % 5 == 0 or replicate == n_replicates - 1:
                elapsed = time.time() - start_time
                avg_time = elapsed / current_run
                remaining = (total_runs - current_run) * avg_time
                
                print(f"  Replicate {replicate + 1}/{n_replicates} "
                      f"[Run {current_run}/{total_runs}] "
                      f"- Est. remaining: {remaining/60:.1f}min")
    
    # Save results
    output_file = output_path / f'experiment_results_{int(time.time())}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for result in all_results:
        serializable = {
            'p': result['p'],
            'replicate': result['replicate'],
            'final_stats': result['final_stats'],
            'parameters': result['parameters'],
            'history': {k: [float(v) for v in vals] 
                       for k, vals in result['history'].items()}
        }
        # include lineage info if present
        if 'lineage' in result:
            serializable['lineage'] = result['lineage']
        serializable_results.append(serializable)
    
    with open(output_file, 'w') as f:
        json.dump({
            'results': serializable_results,
            'experiment_info': {
                'model': 'NKp',
                'p_levels': p_levels,
                'n_replicates': n_replicates,
                'genome_length': genome_length,
                'K': K,
                'generations': generations,
                'total_time_seconds': time.time() - start_time
            }
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return all_results, output_file


if __name__ == "__main__":
    # Run NKp experiment with default parameters
    # Adjust parameters as needed for your computational resources
    
    results, output_file = run_full_experiment(
        p_levels=[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
        n_replicates=20,  # Increase for better statistics
        genome_length=50,
        K=2,  # Start with K=2 for moderate epistasis
        generations=500,
        output_dir='results'
    )
    
    print(f"\nTo analyze results, run: python analyze_results.py {output_file}")
