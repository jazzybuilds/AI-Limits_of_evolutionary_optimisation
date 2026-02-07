"""
Compare short vs long run results to check equilibrium.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def load_and_aggregate(filename):
    """Load results and compute mean diversity by p."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = {}
    for result in data['results']:
        p = result.get('p', result.get('neutrality', 0.0))
        if p not in results:
            results[p] = []
        
        # Get final hamming distance
        final_hamming = result['history']['hamming_distance'][-1]
        results[p].append(final_hamming)
    
    # Compute means and std errors
    summary = {}
    for p in sorted(results.keys()):
        summary[p] = {
            'mean': np.mean(results[p]),
            'se': np.std(results[p]) / np.sqrt(len(results[p]))
        }
    
    return summary, data['experiment_info']['generations']

def compare_results(short_file, long_file):
    """Compare short and long run results."""
    short_results, short_gens = load_and_aggregate(short_file)
    long_results, long_gens = load_and_aggregate(long_file)
    
    print("\n" + "="*70)
    print("EQUILIBRIUM COMPARISON")
    print("="*70)
    print(f"\nShort run: {short_gens} generations")
    print(f"Long run: {long_gens} generations")
    print("\nFinal Hamming Distance by p:")
    print("-"*70)
    print(f"{'p':>6} | {'Short Run':>15} | {'Long Run':>15} | {'Change':>10}")
    print("-"*70)
    
    all_p = sorted(set(list(short_results.keys()) + list(long_results.keys())))
    
    for p in all_p:
        if p in short_results and p in long_results:
            short_mean = short_results[p]['mean']
            long_mean = long_results[p]['mean']
            change = long_mean - short_mean
            change_pct = (change / short_mean) * 100 if short_mean > 0 else 0
            
            print(f"{p:>6.2f} | {short_mean:>8.3f} ± {short_results[p]['se']:>4.3f} | "
                  f"{long_mean:>8.3f} ± {long_results[p]['se']:>4.3f} | "
                  f"{change_pct:>+7.1f}%")
    
    print("-"*70)
    
    # Check if ordering changed
    print("\nDiversity Ranking:")
    short_ranked = sorted(short_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    long_ranked = sorted(long_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"Short run: {' > '.join([f'p={p:.2f}' for p, _ in short_ranked])}")
    print(f"Long run:  {' > '.join([f'p={p:.2f}' for p, _ in long_ranked])}")
    
    if [p for p, _ in short_ranked] == [p for p, _ in long_ranked]:
        print("\n✓ PATTERN CONFIRMED: Diversity ordering remains the same")
    else:
        print("\n✗ PATTERN CHANGED: Different diversity ordering at equilibrium")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    import glob
    
    # Find most recent files
    all_results = sorted(glob.glob('results/experiment_results_*.json'))
    
    if len(all_results) >= 2:
        # Assume second-to-last is short, last is long
        short_file = all_results[-2]
        long_file = all_results[-1]
        
        print(f"Comparing:")
        print(f"  Short: {short_file}")
        print(f"  Long:  {long_file}")
        
        compare_results(short_file, long_file)
    else:
        print("Need at least 2 result files to compare")
        print("Waiting for equilibrium test to complete...")
