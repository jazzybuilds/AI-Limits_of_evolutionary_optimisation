"""
Analysis and visualization of experimental results.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy import stats


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def aggregate_by_neutrality(results_data):
    """
    Aggregate results across replicates for each neutrality level (p).
    
    Returns:
        Dictionary mapping p levels to aggregated statistics
    """
    results = results_data['results']
    aggregated = {}
    
    for result in results:
        # Support both old 'neutrality' key and new 'p' key
        p = result.get('p', result.get('neutrality', 0.0))
        
        if p not in aggregated:
            aggregated[p] = {
                'hamming_distance': [],
                'genetic_entropy': [],
                'unique_genotypes': [],
                'mean_fitness': [],
                'max_fitness': [],
                'history_hamming': [],
                'history_entropy': [],
                'history_fitness': []
            }
        
        # Collect final statistics
        stats = result['final_stats']
        aggregated[p]['hamming_distance'].append(stats['hamming_distance'])
        aggregated[p]['genetic_entropy'].append(stats['genetic_entropy'])
        aggregated[p]['unique_genotypes'].append(stats['unique_genotypes'])
        aggregated[p]['mean_fitness'].append(stats['mean_fitness'])
        aggregated[p]['max_fitness'].append(stats['max_fitness'])
        
        # Collect time series
        history = result['history']
        aggregated[p]['history_hamming'].append(history['hamming_distance'])
        aggregated[p]['history_entropy'].append(history['genetic_entropy'])
        aggregated[p]['history_fitness'].append(history['mean_fitness'])
    
    return aggregated


def compute_summary_statistics(aggregated):
    """
    Compute mean and standard error for each neutrality level (p).
    
    Returns:
        Lists of p levels, means, and standard errors for each metric
    """
    p_levels = sorted(aggregated.keys())
    
    summary = {
        'p': p_levels,
        'hamming_mean': [],
        'hamming_se': [],
        'entropy_mean': [],
        'entropy_se': [],
        'unique_mean': [],
        'unique_se': [],
        'fitness_mean': [],
        'fitness_se': []
    }
    
    for p in p_levels:
        data = aggregated[p]
        
        # Hamming distance
        summary['hamming_mean'].append(np.mean(data['hamming_distance']))
        summary['hamming_se'].append(stats.sem(data['hamming_distance']))
        
        # Genetic entropy
        summary['entropy_mean'].append(np.mean(data['genetic_entropy']))
        summary['entropy_se'].append(stats.sem(data['genetic_entropy']))
        
        # Unique genotypes
        summary['unique_mean'].append(np.mean(data['unique_genotypes']))
        summary['unique_se'].append(stats.sem(data['unique_genotypes']))
        
        # Fitness
        summary['fitness_mean'].append(np.mean(data['mean_fitness']))
        summary['fitness_se'].append(stats.sem(data['mean_fitness']))
    
    return summary


def plot_diversity_vs_neutrality(summary, output_dir='results'):
    """
    Create main results plot: diversity metrics vs neutrality (p).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect of Neutrality (p) on Population Diversity - NKp Model', fontsize=14, fontweight='bold')
    
    p = summary['p']
    p_pct = [n * 100 for n in p]
    
    # Plot 1: Hamming Distance
    ax = axes[0, 0]
    ax.errorbar(p_pct, summary['hamming_mean'], 
                yerr=summary['hamming_se'], 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Avg Pairwise Hamming Distance', fontsize=11)
    ax.set_title('Genetic Diversity (Hamming Distance)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Genetic Entropy
    ax = axes[0, 1]
    ax.errorbar(p_pct, summary['entropy_mean'], 
                yerr=summary['entropy_se'], 
                marker='s', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Genetic Entropy', fontsize=11)
    ax.set_title('Allelic Diversity (Entropy)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Unique Genotypes
    ax = axes[1, 0]
    ax.errorbar(p_pct, summary['unique_mean'], 
                yerr=summary['unique_se'], 
                marker='^', capsize=5, linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Unique Genotypes', fontsize=11)
    ax.set_title('Genotypic Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean Fitness
    ax = axes[1, 1]
    ax.errorbar(p_pct, summary['fitness_mean'], 
                yerr=summary['fitness_se'], 
                marker='d', capsize=5, linewidth=2, markersize=8, color='red')
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Mean Fitness', fontsize=11)
    ax.set_title('Population Fitness (Control)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'diversity_vs_neutrality.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    plt.show()


def plot_time_series(aggregated, output_dir='results'):
    """
    Plot diversity dynamics over time for different neutrality levels (p).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Diversity Dynamics Over Time - NKp Model', fontsize=14, fontweight='bold')
    
    p_levels = sorted(aggregated.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(p_levels)))
    
    for i, p in enumerate(p_levels):
        data = aggregated[p]
        
        # Get time series (average across replicates)
        history_hamming = np.array(data['history_hamming'])
        history_entropy = np.array(data['history_entropy'])
        history_fitness = np.array(data['history_fitness'])
        
        mean_hamming = np.mean(history_hamming, axis=0)
        mean_entropy = np.mean(history_entropy, axis=0)
        mean_fitness = np.mean(history_fitness, axis=0)
        
        generations = range(len(mean_hamming))
        
        label = f'p={p:.2f}'
        
        # Plot Hamming distance over time
        axes[0].plot(generations, mean_hamming, 
                    label=label, color=colors[i], linewidth=2)
        
        # Plot entropy over time
        axes[1].plot(generations, mean_entropy, 
                    label=label, color=colors[i], linewidth=2)
        
        # Plot fitness over time
        axes[2].plot(generations, mean_fitness, 
                    label=label, color=colors[i], linewidth=2)
    
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Hamming Distance')
    axes[0].set_title('Hamming Distance Dynamics')
    axes[0].legend(title='Neutrality (p)', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Genetic Entropy')
    axes[1].set_title('Entropy Dynamics')
    axes[1].legend(title='Neutrality (p)', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Mean Fitness')
    axes[2].set_title('Fitness Dynamics')
    axes[2].legend(title='Neutrality (p)', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'time_series.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    plt.show()


def print_statistical_summary(summary):
    """Print key findings in text format."""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY - NKp Model")
    print("="*70)
    
    p = summary['p']
    
    # Find p level with maximum diversity
    max_hamming_idx = np.argmax(summary['hamming_mean'])
    max_entropy_idx = np.argmax(summary['entropy_mean'])
    
    print(f"\nMaximum Hamming Distance:")
    print(f"  p = {p[max_hamming_idx]:.2f} ({p[max_hamming_idx]*100:.0f}% neutral mutations)")
    print(f"  Value: {summary['hamming_mean'][max_hamming_idx]:.3f} ± "
          f"{summary['hamming_se'][max_hamming_idx]:.3f}")
    
    print(f"\nMaximum Genetic Entropy:")
    print(f"  p = {p[max_entropy_idx]:.2f} ({p[max_entropy_idx]*100:.0f}% neutral mutations)")
    print(f"  Value: {summary['entropy_mean'][max_entropy_idx]:.3f} ± "
          f"{summary['entropy_se'][max_entropy_idx]:.3f}")
    
    print(f"\nComparison (p=0 vs p={p[max_hamming_idx]:.2f}):")
    hamming_increase = ((summary['hamming_mean'][max_hamming_idx] / 
                        summary['hamming_mean'][0]) - 1) * 100
    print(f"  Hamming distance increase: {hamming_increase:.1f}%")
    
    entropy_increase = ((summary['entropy_mean'][max_entropy_idx] / 
                        summary['entropy_mean'][0]) - 1) * 100
    print(f"  Entropy increase: {entropy_increase:.1f}%")
    
    print("\n" + "="*70)


def main(results_file):
    """Main analysis pipeline."""
    print(f"Loading results from: {results_file}")
    
    # Load data
    data = load_results(results_file)
    
    print(f"Loaded {len(data['results'])} experimental runs")
    print(f"Experiment info:")
    for key, value in data['experiment_info'].items():
        print(f"  {key}: {value}")
    
    # Aggregate results
    print("\nAggregating results by neutrality level...")
    aggregated = aggregate_by_neutrality(data)
    
    # Compute summary statistics
    summary = compute_summary_statistics(aggregated)
    
    # Print statistical summary
    print_statistical_summary(summary)
    
    # Create output directory
    output_dir = Path(results_file).parent
    
    # Generate plots
    print("\nGenerating plots...")
    plot_diversity_vs_neutrality(summary, output_dir)
    plot_time_series(aggregated, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("\nLooking for most recent results file...")
        
        results_dir = Path('results')
        if results_dir.exists():
            json_files = list(results_dir.glob('experiment_results_*.json'))
            if json_files:
                latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"Found: {latest_file}")
                main(str(latest_file))
            else:
                print("No results files found. Run experiment first.")
        else:
            print("Results directory not found. Run experiment first.")
    else:
        main(sys.argv[1])
