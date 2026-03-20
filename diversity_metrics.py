"""
Population diversity measurement tools.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


def hamming_distance(genome1, genome2):
    """Calculate Hamming distance between two binary genomes."""
    return np.sum(genome1 != genome2)


def average_pairwise_hamming(population):
    """
    Calculate average pairwise Hamming distance across population.
    Higher values indicate more diversity.
    
    Args:
        population: List or array of binary genomes
        
    Returns:
        Average Hamming distance (normalized by genome length)
    """
    if len(population) < 2:
        return 0.0
    
    population_array = np.array(population)
    
    # Calculate all pairwise Hamming distances
    distances = pdist(population_array, metric='hamming')
    
    # hamming metric returns fraction of differing positions
    # multiply by genome length for actual distance
    genome_length = population_array.shape[1]
    
    return np.mean(distances) * genome_length


def genetic_entropy(population):
    """
    Calculate Shannon entropy at each locus, then average.
    Higher values indicate more diversity.
    
    Args:
        population: List or array of binary genomes
        
    Returns:
        Average entropy across all loci (0 to 1, where 1 is max diversity)
    """
    if len(population) < 2:
        return 0.0
    arr = np.array(population, dtype=np.float32)
    p = arr.mean(axis=0)          # frequency of 1s at each locus, shape (L,)
    q = 1.0 - p
    with np.errstate(divide='ignore', invalid='ignore'):
        H = (np.where(p > 0, -p * np.log2(p), 0.0)
             + np.where(q > 0, -q * np.log2(q), 0.0))
    return float(np.mean(H))


def unique_genotypes(population):
    """Count number of unique genotypes in population."""
    return len({g.tobytes() for g in population})


def calculate_all_metrics(population):
    """Calculate all diversity metrics for a population."""
    return {
        'hamming_distance': average_pairwise_hamming(population),
        'genetic_entropy': genetic_entropy(population),
        'unique_genotypes': unique_genotypes(population),
        'population_size': len(population)
    }
