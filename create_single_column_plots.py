"""
Create single-column plots suitable for 2-column paper format.
Each plot is standalone with larger fonts for readability.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

# Load data
with open('results/experiment_results_1767414768.json', 'r') as f:
    data_long = json.load(f)

# Aggregate by p value
p_values = sorted(set([r['p'] for r in data_long['results']]))
stats = {p: {'hamming': [], 'entropy': [], 'unique': [], 'fitness': [], 'histories': []} 
         for p in p_values}

for result in data_long['results']:
    p = result['p']
    stats[p]['hamming'].append(result['final_stats']['hamming_distance'])
    stats[p]['entropy'].append(result['final_stats']['genetic_entropy'])
    stats[p]['unique'].append(result['final_stats']['unique_genotypes'])
    stats[p]['fitness'].append(result['final_stats']['mean_fitness'])
    stats[p]['histories'].append(result['history'])

# Compute means and SEMs
means = {metric: [] for metric in ['hamming', 'entropy', 'unique', 'fitness']}
sems = {metric: [] for metric in ['hamming', 'entropy', 'unique', 'fitness']}

for p in p_values:
    for metric in means.keys():
        means[metric].append(np.mean(stats[p][metric]))
        sems[metric].append(np.std(stats[p][metric]) / np.sqrt(len(stats[p][metric])))

# ============================================================================
# FIGURE 1: Hamming Distance vs Neutrality
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.errorbar(p_values, means['hamming'], yerr=sems['hamming'],
            marker='o', color='steelblue', capsize=5, linewidth=2, markersize=8,
            markeredgecolor='darkblue', markeredgewidth=1.5)
ax.set_xlabel('Neutrality probability (p)', fontweight='bold')
ax.set_ylabel('Hamming distance', fontweight='bold')
ax.set_title('Genetic Diversity vs Neutrality', fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')
ax.axvline(x=0.75, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
ax.text(0.75, ax.get_ylim()[1]*0.95, 'p=0.75\noptimum', 
        ha='center', va='top', fontsize=9, color='green', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='green'))
plt.tight_layout()
plt.savefig('results/fig1_diversity_vs_neutrality.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig1_diversity_vs_neutrality.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig1_diversity_vs_neutrality.png/pdf")

# ============================================================================
# FIGURE 2: Genetic Entropy vs Neutrality
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.errorbar(p_values, means['entropy'], yerr=sems['entropy'],
            marker='s', color='coral', capsize=5, linewidth=2, markersize=8,
            markeredgecolor='darkred', markeredgewidth=1.5)
ax.set_xlabel('Neutrality probability (p)', fontweight='bold')
ax.set_ylabel('Genetic entropy', fontweight='bold')
ax.set_title('Genetic Entropy vs Neutrality', fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')
ax.axvline(x=0.75, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
plt.tight_layout()
plt.savefig('results/fig2_entropy_vs_neutrality.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig2_entropy_vs_neutrality.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig2_entropy_vs_neutrality.png/pdf")

# ============================================================================
# FIGURE 3: Unique Genotypes vs Neutrality
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.errorbar(p_values, means['unique'], yerr=sems['unique'],
            marker='D', color='teal', capsize=5, linewidth=2, markersize=8,
            markeredgecolor='darkslategray', markeredgewidth=1.5)
ax.set_xlabel('Neutrality probability (p)', fontweight='bold')
ax.set_ylabel('Unique genotypes', fontweight='bold')
ax.set_title('Unique Genotypes vs Neutrality', fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')
ax.axvline(x=0.75, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
plt.tight_layout()
plt.savefig('results/fig3_unique_genotypes.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig3_unique_genotypes.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig3_unique_genotypes.png/pdf")

# ============================================================================
# FIGURE 4: Mean Fitness vs Neutrality
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.errorbar(p_values, means['fitness'], yerr=sems['fitness'],
            marker='^', color='purple', capsize=5, linewidth=2, markersize=8,
            markeredgecolor='indigo', markeredgewidth=1.5)
ax.set_xlabel('Neutrality probability (p)', fontweight='bold')
ax.set_ylabel('Mean fitness', fontweight='bold')
ax.set_title('Fitness Viability Across Neutrality', fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim([0.65, 0.75])
plt.tight_layout()
plt.savefig('results/fig4_fitness_vs_neutrality.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig4_fitness_vs_neutrality.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig4_fitness_vs_neutrality.png/pdf")

# ============================================================================
# FIGURE 5: Temporal Dynamics - Hamming Distance
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))
colors = {'0.0': 'gray', '0.45': 'blue', '0.75': 'orange', '0.9': 'gold'}
labels = {'0.0': 'p=0.0', '0.45': 'p=0.45', '0.75': 'p=0.75', '0.9': 'p=0.9'}

for p in [0.0, 0.45, 0.75, 0.9]:
    histories = stats[p]['histories']
    hamming_over_time = np.array([h['hamming_distance'] for h in histories])
    mean_trajectory = np.mean(hamming_over_time, axis=0)
    sem_trajectory = np.std(hamming_over_time, axis=0) / np.sqrt(len(histories))
    
    generations = np.arange(0, len(mean_trajectory) * 10, 10)
    
    ax.plot(generations, mean_trajectory, label=labels[str(p)], 
            color=colors[str(p)], linewidth=2.5)
    ax.fill_between(generations, 
                     mean_trajectory - sem_trajectory,
                     mean_trajectory + sem_trajectory,
                     alpha=0.2, color=colors[str(p)])

ax.set_xlabel('Generation', fontweight='bold')
ax.set_ylabel('Hamming distance', fontweight='bold')
ax.set_title('Diversity Dynamics Over Time', fontweight='bold', pad=10)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.axvline(x=500, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(500, ax.get_ylim()[1]*0.05, 'Short-term\nendpoint', 
        ha='right', va='bottom', fontsize=8, color='red')
plt.tight_layout()
plt.savefig('results/fig5_hamming_temporal.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig5_hamming_temporal.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig5_hamming_temporal.png/pdf")

# ============================================================================
# FIGURE 6: Temporal Dynamics - Genetic Entropy
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))

for p in [0.0, 0.45, 0.75, 0.9]:
    histories = stats[p]['histories']
    entropy_over_time = np.array([h['genetic_entropy'] for h in histories])
    mean_trajectory = np.mean(entropy_over_time, axis=0)
    sem_trajectory = np.std(entropy_over_time, axis=0) / np.sqrt(len(histories))
    
    generations = np.arange(0, len(mean_trajectory) * 10, 10)
    
    ax.plot(generations, mean_trajectory, label=labels[str(p)], 
            color=colors[str(p)], linewidth=2.5)
    ax.fill_between(generations, 
                     mean_trajectory - sem_trajectory,
                     mean_trajectory + sem_trajectory,
                     alpha=0.2, color=colors[str(p)])

ax.set_xlabel('Generation', fontweight='bold')
ax.set_ylabel('Genetic entropy', fontweight='bold')
ax.set_title('Entropy Dynamics Over Time', fontweight='bold', pad=10)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.axvline(x=500, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
plt.tight_layout()
plt.savefig('results/fig6_entropy_temporal.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig6_entropy_temporal.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig6_entropy_temporal.png/pdf")

# ============================================================================
# FIGURE 7: Temporal Dynamics - Mean Fitness
# ============================================================================
fig, ax = plt.subplots(figsize=(3.5, 3))

for p in [0.0, 0.45, 0.75, 0.9]:
    histories = stats[p]['histories']
    fitness_over_time = np.array([h['mean_fitness'] for h in histories])
    mean_trajectory = np.mean(fitness_over_time, axis=0)
    sem_trajectory = np.std(fitness_over_time, axis=0) / np.sqrt(len(histories))
    
    generations = np.arange(0, len(mean_trajectory) * 10, 10)
    
    ax.plot(generations, mean_trajectory, label=labels[str(p)], 
            color=colors[str(p)], linewidth=2.5)
    ax.fill_between(generations, 
                     mean_trajectory - sem_trajectory,
                     mean_trajectory + sem_trajectory,
                     alpha=0.2, color=colors[str(p)])

ax.set_xlabel('Generation', fontweight='bold')
ax.set_ylabel('Mean fitness', fontweight='bold')
ax.set_title('Fitness Dynamics Over Time', fontweight='bold', pad=10)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/fig7_fitness_temporal.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig7_fitness_temporal.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig7_fitness_temporal.png/pdf")

# ============================================================================
# FIGURE 8: Equilibrium Comparison - Side by Side
# ============================================================================
# Load short-term data
with open('results/experiment_results_1767413931.json', 'r') as f:
    data_short = json.load(f)

stats_short = {p: [] for p in [0.0, 0.45, 0.75, 0.9]}
for result in data_short['results']:
    p = result['p']
    if p in stats_short:
        stats_short[p].append(result['history']['hamming_distance'][-1])

p_common = [0.0, 0.45, 0.75, 0.9]
mean_short = [np.mean(stats_short[p]) for p in p_common]
sem_short = [np.std(stats_short[p])/np.sqrt(len(stats_short[p])) for p in p_common]
mean_long = [np.mean(stats[p]['hamming']) for p in p_common]
sem_long = [np.std(stats[p]['hamming'])/np.sqrt(len(stats[p]['hamming'])) for p in p_common]

fig, ax = plt.subplots(figsize=(3.5, 3))
x = np.arange(len(p_common))
width = 0.35

bars1 = ax.bar(x - width/2, mean_short, width, yerr=sem_short,
               label='500 gen', color='steelblue', alpha=0.7, capsize=4,
               edgecolor='darkblue', linewidth=1.5)
bars2 = ax.bar(x + width/2, mean_long, width, yerr=sem_long,
               label='1500 gen', color='orangered', alpha=0.7, capsize=4,
               edgecolor='darkred', linewidth=1.5)

ax.set_xlabel('Neutrality probability (p)', fontweight='bold')
ax.set_ylabel('Hamming distance', fontweight='bold')
ax.set_title('Equilibrium Comparison', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels([f'{p:.2f}' for p in p_common])
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--', axis='y')

# Highlight p=0.75
ax.axvspan(x[2]-0.4, x[2]+0.4, alpha=0.1, color='green')

plt.tight_layout()
plt.savefig('results/fig8_equilibrium_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig8_equilibrium_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig8_equilibrium_comparison.png/pdf")

# ============================================================================
# FIGURE 9: Change from 500 to 1500 generations
# ============================================================================
changes = [mean_long[i] - mean_short[i] for i in range(len(p_common))]
change_errs = [np.sqrt(sem_short[i]**2 + sem_long[i]**2) for i in range(len(p_common))]

fig, ax = plt.subplots(figsize=(3.5, 3))
colors_bars = ['gray' if p != 0.75 else 'green' for p in p_common]
bars = ax.bar(range(len(p_common)), changes, yerr=change_errs,
              color=colors_bars, alpha=0.7, capsize=5,
              edgecolor='black', linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Neutrality probability (p)', fontweight='bold')
ax.set_ylabel('Change in Hamming distance', fontweight='bold')
ax.set_title('Diversity Growth (500→1500 gen)', fontweight='bold', pad=10)
ax.set_xticks(range(len(p_common)))
ax.set_xticklabels([f'{p:.2f}' for p in p_common])
ax.grid(alpha=0.3, linestyle='--', axis='y')

# Add percentage labels
for i, (change, err) in enumerate(zip(changes, change_errs)):
    if mean_short[i] > 0:
        pct = (change / mean_short[i]) * 100
        ax.text(i, change + err + 0.02, f'+{pct:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/fig9_equilibrium_change.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig9_equilibrium_change.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Created: fig9_equilibrium_change.png/pdf")

print("\n" + "="*70)
print("All 9 figures saved in PNG and PDF formats with 300 dpi")
print("Single-column width (3.5 inches) suitable for 2-column papers")
print("="*70)
