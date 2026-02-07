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
    population_array = np.array(population)
    genome_length = population_array.shape[1]
    
    entropies = []
    for locus in range(genome_length):
        # Get allele frequencies at this locus
        allele_counts = np.bincount(population_array[:, locus], minlength=2)
        frequencies = allele_counts / len(population)
        
        # Calculate Shannon entropy
        # H = -sum(p * log2(p)) for p > 0
        entropy = 0.0
        for freq in frequencies:
            if freq > 0:
                entropy -= freq * np.log2(freq)
        
        entropies.append(entropy)
    
    # Return average entropy (normalized to 0-1, where 1 is max entropy for binary)
    return np.mean(entropies)


def unique_genotypes(population):
    """Count number of unique genotypes in population."""
    unique = set(tuple(genome) for genome in population)
    return len(unique)


def calculate_all_metrics(population):
    """Calculate all diversity metrics for a population."""
    return {
        'hamming_distance': average_pairwise_hamming(population),
        'genetic_entropy': genetic_entropy(population),
        'unique_genotypes': unique_genotypes(population),
        'population_size': len(population)
    }
