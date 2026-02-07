"""
Test if diversity patterns hold at longer timescales.
"""
from run_experiment import run_full_experiment

if __name__ == "__main__":
    print("Running equilibrium test with 1500 generations...")
    print("Testing key p values: 0.0, 0.45, 0.75, 0.9")
    print("This will take ~10-15 minutes\n")
    
    results, output_file = run_full_experiment(
        p_levels=[0.0, 0.45, 0.75, 0.9],  # Just test key levels
        n_replicates=15,  # Fewer replicates for speed
        genome_length=50,
        K=2,
        generations=1500,  # 3x longer
        output_dir='results'
    )
    
    print(f"\nResults saved to: {output_file}")
    print("Run: python analyze_results.py {0}".format(output_file))
