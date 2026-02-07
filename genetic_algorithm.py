"""
Genetic Algorithm with Neutral Evolution
Implements asexual reproduction with bit-flip mutations.
"""
import numpy as np
from diversity_metrics import calculate_all_metrics


class GeneticAlgorithm:
    """
    Simple genetic algorithm for studying neutral evolution.
    
    Features:
    - Fixed-length binary genomes
    - Asexual reproduction
    - Bit-flip mutation
    - Tournament selection
    """
    
    def __init__(self, fitness_landscape, population_size=100, 
                 mutation_rate=0.01, tournament_size=3, seed=None,
                 lineage_tracking=False):
        """
        Args:
            fitness_landscape: NKLandscape object for fitness evaluation
            population_size: Number of individuals
            mutation_rate: Probability of bit flip per gene
            tournament_size: Number of individuals in tournament selection
            seed: Random seed for reproducibility
        """
        self.landscape = fitness_landscape
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.rng = np.random.RandomState(seed)
        
        self.genome_length = fitness_landscape.N
        self.population = None
        self.fitness_cache = {}
        
        # Statistics tracking
        self.generation = 0
        self.history = {
            'generation': [],
            'mean_fitness': [],
            'max_fitness': [],
            'hamming_distance': [],
            'genetic_entropy': [],
            'unique_genotypes': []
        }
        # Lineage tracking (optional)
        self.lineage_tracking = lineage_tracking
        self.next_id = 0
        self.ids = []
        self.parent_log = []  # (child_id, parent_id, generation)
        self.snapshot_history = []
    
    def initialize_population(self):
        """Create random initial population."""
        self.population = [
            self.rng.randint(0, 2, size=self.genome_length)
            for _ in range(self.population_size)
        ]
        self.fitness_cache = {}
        self.generation = 0
        if self.lineage_tracking:
            self.ids = []
            for ind in self.population:
                self.ids.append(self.next_id)
                self.parent_log.append((self.next_id, -1, 0))
                self.next_id += 1
            self._record_snapshot()
    
    def evaluate_fitness(self, genome):
        """
        Evaluate fitness with caching.
        
        Args:
            genome: Binary array
            
        Returns:
            Fitness value
        """
        genome_key = tuple(genome)
        if genome_key not in self.fitness_cache:
            self.fitness_cache[genome_key] = self.landscape.evaluate(genome)
        return self.fitness_cache[genome_key]
    
    def get_population_fitness(self):
        """Return fitness values for entire population."""
        return [self.evaluate_fitness(ind) for ind in self.population]
    
    def tournament_selection(self):
        """
        Select parent using tournament selection.
        
        Returns:
            Selected genome (copy)
        """
        # Randomly select tournament_size individuals
        contestants_idx = self.rng.choice(
            len(self.population), 
            size=self.tournament_size, 
            replace=False
        )
        
        # Evaluate fitness of contestants
        best_idx = None
        best_fitness = -1
        for idx in contestants_idx:
            fitness = self.evaluate_fitness(self.population[idx])
            if fitness > best_fitness:
                best_fitness = fitness
                best_idx = idx
        
        # return genome copy and selected index (for parent tracking)
        return self.population[best_idx].copy(), best_idx
    
    def mutate(self, genome):
        """
        Apply bit-flip mutation.
        
        Args:
            genome: Binary array (modified in place)
            
        Returns:
            Mutated genome
        """
        for i in range(len(genome)):
            if self.rng.random() < self.mutation_rate:
                genome[i] = 1 - genome[i]  # Flip bit
        return genome
    
    def evolve_generation(self):
        """Run one generation of evolution."""
        new_population = []
        new_ids = []

        for _ in range(self.population_size):
            # Select parent (genome, index)
            parent_genome, parent_idx = self.tournament_selection()

            # Create offspring through mutation (asexual reproduction)
            offspring = self.mutate(parent_genome.copy())

            new_population.append(offspring)
            if self.lineage_tracking:
                parent_id = self.ids[parent_idx]
                child_id = self.next_id
                self.parent_log.append((child_id, parent_id, self.generation + 1))
                new_ids.append(child_id)
                self.next_id += 1

        self.population = new_population
        if self.lineage_tracking:
            self.ids = new_ids
        self.generation += 1
    
    def record_statistics(self):
        """Calculate and store population statistics."""
        fitness_values = self.get_population_fitness()
        diversity_metrics = calculate_all_metrics(self.population)
        
        self.history['generation'].append(self.generation)
        self.history['mean_fitness'].append(np.mean(fitness_values))
        self.history['max_fitness'].append(np.max(fitness_values))
        self.history['hamming_distance'].append(diversity_metrics['hamming_distance'])
        self.history['genetic_entropy'].append(diversity_metrics['genetic_entropy'])
        self.history['unique_genotypes'].append(diversity_metrics['unique_genotypes'])
        if self.lineage_tracking:
            self._record_snapshot()


    def _record_snapshot(self):
        """Record current population as snapshot of (id, bitstring)."""
        individuals = []
        for idx, genome in enumerate(self.population):
            gid = self.ids[idx] if idx < len(self.ids) else None
            try:
                bitstr = ''.join(str(int(b)) for b in genome)
            except Exception:
                bitstr = str(tuple(int(b) for b in genome))
            individuals.append((gid, bitstr))

        self.snapshot_history.append({'generation': self.generation, 'individuals': individuals})
    
    def run(self, generations, record_interval=10, verbose=False):
        """
        Run evolution for specified number of generations.
        
        Args:
            generations: Number of generations to evolve
            record_interval: How often to record statistics
            verbose: Print progress updates
            
        Returns:
            Dictionary of historical statistics
        """
        # Record initial state
        self.record_statistics()
        
        for gen in range(generations):
            self.evolve_generation()
            
            # Record statistics at intervals
            if (gen + 1) % record_interval == 0:
                self.record_statistics()
                
                if verbose:
                    fitness_values = self.get_population_fitness()
                    print(f"Gen {self.generation}: "
                          f"Mean Fitness={np.mean(fitness_values):.3f}, "
                          f"Diversity={self.history['hamming_distance'][-1]:.2f}")
        
        return self.history
    
    def get_summary_statistics(self):
        """Get summary of final population state."""
        fitness_values = self.get_population_fitness()
        diversity_metrics = calculate_all_metrics(self.population)
        
        return {
            'generation': self.generation,
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'max_fitness': np.max(fitness_values),
            'min_fitness': np.min(fitness_values),
            **diversity_metrics
        }
