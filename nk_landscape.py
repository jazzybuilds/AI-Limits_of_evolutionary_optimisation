"""
NKp Fitness Landscape Implementation (Barnett et al.)
Provides epistatic fitness evaluation with controllable neutrality.
"""
import numpy as np


class NKLandscape:
    """
    NKp fitness landscape where:
    - N = genome length
    - K = number of epistatic interactions per gene
    - p = probability that a mutation is neutral
    
    Higher K = more rugged landscape (more gene interactions)
    Higher p = more neutrality (neutral networks)
    """
    
    def __init__(self, N, K, p=0.0, seed=None):
        """
        Args:
            N: Genome length
            K: Number of epistatic interactions (0 <= K < N)
            p: Neutrality probability - probability that a mutation is neutral (0.0-1.0)
            seed: Random seed for reproducibility
        """
        if K >= N:
            raise ValueError(f"K must be less than N (K={K}, N={N})")
        
        self.N = N
        self.K = K
        self.p = p
        self.rng = np.random.RandomState(seed)
        
        # For each gene, determine which other genes it interacts with
        self.interactions = {}
        for gene in range(N):
            # Each gene interacts with K random other genes
            available = [g for g in range(N) if g != gene]
            partners = self.rng.choice(available, size=min(K, len(available)), replace=False)
            self.interactions[gene] = np.sort(np.append([gene], partners))
        
        # Create lookup tables for fitness contributions with neutrality
        # With probability p, mutations are neutral (adjacent entries have same fitness)
        self.fitness_tables = {}
        for gene, interaction_set in self.interactions.items():
            table_size = 2 ** len(interaction_set)
            self.fitness_tables[gene] = self._create_neutral_table(table_size)
    
    def _create_neutral_table(self, table_size):
        """
        Create a fitness lookup table with neutrality.
        
        With probability p, a table entry will have the same fitness as 
        a neighboring entry (one bit flip away), creating neutral mutations.
        
        Args:
            table_size: Size of lookup table (2^(K+1))
            
        Returns:
            Array of fitness values with neutral networks
        """
        # Start with random fitness values
        table = self.rng.uniform(0, 1, size=table_size)
        
        # For each entry, with probability p, make it neutral by copying
        # fitness from a random neighboring entry (one bit flip away)
        for idx in range(table_size):
            if self.rng.random() < self.p:
                # Find neighboring indices (one bit flip away)
                neighbors = []
                for bit in range(int(np.log2(table_size))):
                    neighbor = idx ^ (1 << bit)  # Flip bit at position 'bit'
                    neighbors.append(neighbor)
                
                # Copy fitness from a random neighbor to create neutral mutation
                if neighbors:
                    neighbor_idx = self.rng.choice(neighbors)
                    table[idx] = table[neighbor_idx]
        
        return table
    
    def evaluate(self, genome):
        """
        Calculate fitness of a genome.
        
        Args:
            genome: Binary array of length N
            
        Returns:
            Fitness value (0.0 to 1.0)
        """
        if len(genome) != self.N:
            raise ValueError(f"Genome length must be {self.N}, got {len(genome)}")
        
        # Sum fitness contributions from all genes
        total_fitness = 0.0
        for gene in range(self.N):
            # Get the values of this gene and its interaction partners
            interaction_indices = self.interactions[gene]
            interaction_values = genome[interaction_indices]
            
            # Convert binary array to table index
            table_index = int(''.join(map(str, interaction_values)), 2)
            
            # Add this gene's contribution
            total_fitness += self.fitness_tables[gene][table_index]
        
        # Normalize by number of genes
        return total_fitness / self.N
    
    def get_info(self):
        """Return landscape configuration info."""
        return {
            'N': self.N,
            'K': self.K,
            'p': self.p,
            'model': 'NKp'
        }
