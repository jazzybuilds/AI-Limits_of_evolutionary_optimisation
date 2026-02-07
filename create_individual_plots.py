"""
Create individual plots from the same data used for multi-panel plots.
Uses the same plotting logic but saves each panel as a separate file.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def aggregate_by_neutrality(results_data):
    """Aggregate results across replicates for each neutrality level (p)."""
    results = results_data['results']
    aggregated = {}
    
    for result in results:
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
        
        stats_data = result['final_stats']
        aggregated[p]['hamming_distance'].append(stats_data['hamming_distance'])
        aggregated[p]['genetic_entropy'].append(stats_data['genetic_entropy'])
        aggregated[p]['unique_genotypes'].append(stats_data['unique_genotypes'])
        aggregated[p]['mean_fitness'].append(stats_data['mean_fitness'])
        aggregated[p]['max_fitness'].append(stats_data['max_fitness'])
        
        history = result['history']
        aggregated[p]['history_hamming'].append(history['hamming_distance'])
        aggregated[p]['history_entropy'].append(history['genetic_entropy'])
        aggregated[p]['history_fitness'].append(history['mean_fitness'])
    
    return aggregated


def compute_summary_statistics(aggregated):
    """Compute mean and standard error for each neutrality level."""
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
        
        summary['hamming_mean'].append(np.mean(data['hamming_distance']))
        summary['hamming_se'].append(stats.sem(data['hamming_distance']))
        
        summary['entropy_mean'].append(np.mean(data['genetic_entropy']))
        summary['entropy_se'].append(stats.sem(data['genetic_entropy']))
        
        summary['unique_mean'].append(np.mean(data['unique_genotypes']))
        summary['unique_se'].append(stats.sem(data['unique_genotypes']))
        
        summary['fitness_mean'].append(np.mean(data['mean_fitness']))
        summary['fitness_se'].append(stats.sem(data['mean_fitness']))
    
    return summary


def create_individual_diversity_plots(summary, output_dir='results'):
    """Create 4 individual plots from diversity vs neutrality data."""
    
    p = summary['p']
    p_pct = [n * 100 for n in p]
    
    # Panel 1: Hamming Distance
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(p_pct, summary['hamming_mean'], 
                yerr=summary['hamming_se'], 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Avg Pairwise Hamming Distance', fontsize=11)
    ax.set_title('Genetic Diversity (Hamming Distance)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_panel_1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_panel_1.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created diversity_panel_1.png/pdf (Hamming Distance)")
    
    # Panel 2: Genetic Entropy
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(p_pct, summary['entropy_mean'], 
                yerr=summary['entropy_se'], 
                marker='s', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Genetic Entropy', fontsize=11)
    ax.set_title('Allelic Diversity (Entropy)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_panel_2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_panel_2.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created diversity_panel_2.png/pdf (Genetic Entropy)")
    
    # Panel 3: Unique Genotypes
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(p_pct, summary['unique_mean'], 
                yerr=summary['unique_se'], 
                marker='^', capsize=5, linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Unique Genotypes', fontsize=11)
    ax.set_title('Genotypic Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_panel_3.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_panel_3.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created diversity_panel_3.png/pdf (Unique Genotypes)")
    
    # Panel 4: Mean Fitness
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(p_pct, summary['fitness_mean'], 
                yerr=summary['fitness_se'], 
                marker='d', capsize=5, linewidth=2, markersize=8, color='red')
    ax.set_xlabel('p - Neutral Mutation Probability (%)', fontsize=11)
    ax.set_ylabel('Mean Fitness', fontsize=11)
    ax.set_title('Population Fitness (Control)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_panel_4.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/diversity_panel_4.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created diversity_panel_4.png/pdf (Mean Fitness)")


def create_individual_time_series_plots(aggregated, output_dir='results'):
    """Create 3 individual plots from time series data."""
    
    p_levels = sorted(aggregated.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(p_levels)))
    
    # Load short-run (500 gen) aggregated data to overlay for comparison
    short_file = f'{output_dir}/experiment_results_1767413931.json'
    short_agg = None
    try:
        short_data = load_results(short_file)
        short_agg = aggregate_by_neutrality(short_data)
    except Exception:
        short_agg = None

    # Panel 1: Hamming Distance Over Time
    fig, ax = plt.subplots(figsize=(6, 4))
    record_interval = 10  # generations between recorded snapshots
    for i, p in enumerate(p_levels):
        data = aggregated[p]
        history_hamming = np.array(data['history_hamming'])  # shape (n_reps, t)
        # plot thin replicate traces behind mean (optional)
        for rep in history_hamming:
            ax.plot(np.arange(0, len(rep) * record_interval, record_interval), rep,
                    color=colors[i], alpha=0.10, linewidth=0.8)

        mean_hamming = np.mean(history_hamming, axis=0)
        sem_hamming = stats.sem(history_hamming, axis=0, ddof=1)
        generations = np.arange(0, len(mean_hamming) * record_interval, record_interval)

        ax.plot(generations, mean_hamming, label=f'p={p:.2f}', color=colors[i], linewidth=2)
        ax.fill_between(generations, mean_hamming - sem_hamming, mean_hamming + sem_hamming,
                        color=colors[i], alpha=0.25)

        # overlay short-run mean (dashed) if available
        if short_agg is not None and p in short_agg:
            short_hist = np.array(short_agg[p]['history_hamming'])
            mean_short = np.mean(short_hist, axis=0)
            gens_short = np.arange(0, len(mean_short) * record_interval, record_interval)
            ax.plot(gens_short, mean_short, linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Hamming Distance', fontsize=11)
    ax.set_title('Hamming Distance Dynamics', fontweight='bold')
    ax.legend(title='Neutrality (p)', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_panel_1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/time_panel_1.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created time_panel_1.png/pdf (Hamming Dynamics)")
    
    # Panel 2: Entropy Over Time
    fig, ax = plt.subplots(figsize=(6, 4))
    record_interval = 10
    for i, p in enumerate(p_levels):
        data = aggregated[p]
        history_entropy = np.array(data['history_entropy'])
        # replicate traces
        for rep in history_entropy:
            ax.plot(np.arange(0, len(rep) * record_interval, record_interval), rep,
                    color=colors[i], alpha=0.10, linewidth=0.8)

        mean_entropy = np.mean(history_entropy, axis=0)
        sem_entropy = stats.sem(history_entropy, axis=0, ddof=1)
        generations = np.arange(0, len(mean_entropy) * record_interval, record_interval)

        ax.plot(generations, mean_entropy, label=f'p={p:.2f}', color=colors[i], linewidth=2)
        ax.fill_between(generations, mean_entropy - sem_entropy, mean_entropy + sem_entropy,
                        color=colors[i], alpha=0.25)

        # overlay short-run mean (dashed) if available
        if short_agg is not None and p in short_agg:
            short_hist_e = np.array(short_agg[p]['history_entropy'])
            mean_short_e = np.mean(short_hist_e, axis=0)
            gens_short_e = np.arange(0, len(mean_short_e) * record_interval, record_interval)
            ax.plot(gens_short_e, mean_short_e, linestyle='--', color=colors[i], linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Genetic Entropy', fontsize=11)
    ax.set_title('Entropy Dynamics', fontweight='bold')
    ax.legend(title='Neutrality (p)', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_panel_2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/time_panel_2.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created time_panel_2.png/pdf (Entropy Dynamics)")
    
    # Panel 3: Fitness Over Time
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, p in enumerate(p_levels):
        data = aggregated[p]
        history_fitness = np.array(data['history_fitness'])
        mean_fitness = np.mean(history_fitness, axis=0)
        generations = range(len(mean_fitness))
        ax.plot(generations, mean_fitness, label=f'p={p:.2f}', 
                color=colors[i], linewidth=2)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Mean Fitness', fontsize=11)
    ax.set_title('Fitness Dynamics', fontweight='bold')
    ax.legend(title='Neutrality (p)', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_panel_3.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/time_panel_3.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created time_panel_3.png/pdf (Fitness Dynamics)")


def create_individual_equilibrium_plots(output_dir='results'):
    """Create 2 individual plots from equilibrium comparison data."""
    
    # Load both datasets
    short_file = f'{output_dir}/experiment_results_1767413931.json'  # 500 generations
    long_file = f'{output_dir}/experiment_results_1767414768.json'   # 1500 generations
    
    def load_eq_results(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = {}
        for result in data['results']:
            p = result.get('p', result.get('neutrality', 0.0))
            if p not in results:
                results[p] = []
            final_hamming = result['history']['hamming_distance'][-1]
            results[p].append(final_hamming)
        
        p_values = sorted(results.keys())
        means = [np.mean(results[p]) for p in p_values]
        sems = [np.std(results[p]) / np.sqrt(len(results[p])) for p in p_values]
        generations = data['experiment_info']['generations']
        
        return p_values, means, sems, generations
    
    p_short, mean_short, sem_short, gen_short = load_eq_results(short_file)
    p_long, mean_long, sem_long, gen_long = load_eq_results(long_file)
    
    # Panel 1: Side-by-side comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(p_short, mean_short, yerr=sem_short, 
                marker='o', linestyle='-', linewidth=2, markersize=8,
                capsize=5, label=f'{gen_short} generations', color='steelblue', alpha=0.7)
    ax.errorbar(p_long, mean_long, yerr=sem_long,
                marker='s', linestyle='-', linewidth=2, markersize=8,
                capsize=5, label=f'{gen_long} generations', color='orangered', alpha=0.7)
    ax.set_xlabel('Neutrality probability (p)', fontsize=11)
    ax.set_ylabel('Hamming distance', fontsize=11)
    ax.set_title('Equilibrium Emergence: Short vs Extended Evolution', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axvline(x=0.75, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/equilibrium_panel_1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/equilibrium_panel_1.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created equilibrium_panel_1.png/pdf (Equilibrium Comparison)")
    
    # Panel 2: Change from 500 to 1500 generations
    changes = []
    change_errs = []
    p_common = []
    for p in p_short:
        if p in p_long:
            idx_short = p_short.index(p)
            idx_long = p_long.index(p)
            change = mean_long[idx_long] - mean_short[idx_short]
            change_err = np.sqrt(sem_short[idx_short]**2 + sem_long[idx_long]**2)
            changes.append(change)
            change_errs.append(change_err)
            p_common.append(p)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_bar = ['green' if p == 0.75 else 'gray' for p in p_common]
    ax.bar(p_common, changes, yerr=change_errs, color=colors_bar, alpha=0.7, 
           capsize=5, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Neutrality probability (p)', fontsize=11)
    ax.set_ylabel('Change in Hamming distance\n(1500 gen - 500 gen)', fontsize=11)
    ax.set_title('Diversity Change: Evidence for p=0.75 Optimum', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add text annotations for p=0.75 and p=0.9
    for i, p in enumerate(p_common):
        if p in [0.75, 0.9]:
            ax.text(p, changes[i] + change_errs[i] + 0.02, 
                   f'+{changes[i]:.2f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/equilibrium_panel_2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/equilibrium_panel_2.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created equilibrium_panel_2.png/pdf (Equilibrium Change)")


def main():
    """Generate all individual panel plots."""
    output_dir = 'results'
    
    print("Creating individual plots from experimental data...")
    print("="*60)
    
    # Load 1500 generation data for diversity plots
    data = load_results(f'{output_dir}/experiment_results_1767414768.json')
    aggregated = aggregate_by_neutrality(data)
    summary = compute_summary_statistics(aggregated)
    
    print("\n1. Creating diversity panels (4 plots)...")
    create_individual_diversity_plots(summary, output_dir)
    
    print("\n2. Creating time series panels (3 plots)...")
    create_individual_time_series_plots(aggregated, output_dir)
    
    print("\n3. Creating equilibrium panels (2 plots)...")
    create_individual_equilibrium_plots(output_dir)
    
    print("\n" + "="*60)
    print("All individual plots created successfully!")
    print("\nTotal: 9 plots (PNG + PDF for each)")


if __name__ == '__main__':
    main()
