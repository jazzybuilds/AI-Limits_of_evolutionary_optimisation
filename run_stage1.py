"""
Stage 1 Neuroevolution Experiment Runner
========================================
GA (with neutrality + epistasis) evolves CTRNN or Feedforward NN agents
on MiniGrid or CartPole.

Fitness = w_perf * train_performance + w_gen * generalisation_performance

Learnability is parked until Stage 2.

All hyperparameters are at the top of this file for easy editing.
"""
import json
import time
import numpy as np
from pathlib import Path

# ============================================================
#  HYPERPARAMETERS — edit these before running
# ============================================================

CONFIG = {

    # ---- Environment --------------------------------------------------------
    # Choose: 'cartpole' | 'acrobot' | 'lunarlander' | 'lunarlander_continuous' | 'mountaincar' | 'mountaincarcontinuous' | 'minigrid'
    "ENV_TYPE": "lunarlander_continuous",  # <- switch here

    # MiniGrid: list of training environments (agent is evaluated on all)
    "MINIGRID_TRAIN_ENVS": [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-8x8-v0",
    ],

    # MiniGrid: held-out environments for generalisation test
    "MINIGRID_TEST_ENVS": [
        "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-FourRooms-v0",
    ],

    # CartPole: physics parameter overrides for training envs.
    # Each dict can override: length, masspole, masscart, gravity, force_mag.
    # {} means standard CartPole-v1 params.
    "CARTPOLE_TRAIN_PARAMS": [
        {},                          # standard: length=0.5, gravity=9.8
        {"length": 0.75},            # longer pole
    ],

    # CartPole: held-out physics configs for generalisation test.
    # Agent never sees these during training.
    "CARTPOLE_TEST_PARAMS": [
        {"length": 0.25},            # short pole (faster dynamics)
        {"length": 1.0},             # very long pole (slower dynamics)
        {"gravity": 7.0},            # low gravity
        {"gravity": 12.0},           # high gravity
        {"masspole": 0.3},           # heavy pole
    ],

    # Acrobot: physics parameter overrides (see neuroevo_fitness._make_acrobot)
    # Vary link lengths and masses — changes moment of inertia and swing energy.
    "ACROBOT_TRAIN_PARAMS": [
        {},                                      # standard
        {"LINK_LENGTH_2": 0.75},                 # shorter inner link
    ],
    "ACROBOT_TEST_PARAMS": [
        {"LINK_LENGTH_2": 1.5},                  # longer outer link
        {"LINK_LENGTH_1": 1.5},                  # longer inner link
        {"LINK_MASS_1": 1.5, "LINK_MASS_2": 1.5},  # heavier links
        {"LINK_MOI": 2.0},                       # higher moment of inertia
    ],

    # LunarLander (discrete): kwargs passed directly to gym.make('LunarLander-v3')
    "LUNARLANDER_TRAIN_PARAMS": [
        {},                                      # standard (no wind)
        {"wind_power": 5.0, "enable_wind": True},
    ],
    "LUNARLANDER_TEST_PARAMS": [
        {"gravity": -7.0},                       # low gravity
        {"gravity": -11.5},                      # high gravity
        {"wind_power": 15.0, "turbulence_power": 1.5, "enable_wind": True},
    ],

    # LunarLanderContinuous: same physics kwargs, Box(2,) action space
    "LUNARLANDER_CONTINUOUS_TRAIN_PARAMS": [
        {},                                      # standard (no wind)
        {"wind_power": 5.0, "enable_wind": True},
    ],
    "LUNARLANDER_CONTINUOUS_TEST_PARAMS": [
        {"gravity": -7.0},                       # low gravity
        {"gravity": -11.5},                      # high gravity
        {"wind_power": 15.0, "turbulence_power": 1.5, "enable_wind": True},
    ],

    # MountainCar (discrete): attribute overrides on env.unwrapped
    "MOUNTAINCAR_TRAIN_PARAMS": [
        {},                                      # standard
        {"power": 0.0015},                       # stronger engine
    ],
    "MOUNTAINCAR_TEST_PARAMS": [
        {"power": 0.0008},                       # weaker engine
        {"goal_position": 0.35},                 # lower goal
        {"min_position": -1.5},                  # deeper valley
    ],

    # MountainCarContinuous: same physics params, continuous force output
    "MOUNTAINCARCONTINUOUS_TRAIN_PARAMS": [
        {},                                      # standard
        {"power": 0.0015},                       # stronger engine
    ],
    "MOUNTAINCARCONTINUOUS_TEST_PARAMS": [
        {"power": 0.0008},                       # weaker engine — harder
        {"power": 0.0012},                       # stronger engine — different dynamics
        {"goal_position": 0.50},                 # higher goal — harder than training 0.45
        {"max_speed": 0.05},                     # slower max speed — tighter control needed
    ],
    "MOUNTAINCARCONTINUOUS_MAX_STEPS": 999,      # gym default is 999

    # Seeds used during training evaluation (in-distribution)
    "TRAIN_SEEDS": [0, 1, 2],

    # Seeds used during generalisation evaluation (unseen)
    "TEST_SEEDS": [10, 11, 12],

    # Number of evaluation episodes per seed per environment
    "N_EVAL_EPISODES": 2,

    # Max steps per episode (CartPole max=500; MiniGrid suggest 200)
    "MAX_STEPS_PER_EPISODE": 500,    # CartPole / LunarLander / MiniGrid
    "ACROBOT_MAX_STEPS":      200,    # Acrobot only terminates on success; 200 steps ≈ CartPole timing
    "MOUNTAINCAR_MAX_STEPS":  200,    # same reason as Acrobot

    # ---- Controller ---------------------------------------------------------
    # Choose: 'feedforward' or 'ctrnn'
    "CONTROLLER_TYPE": "ctrnn",

    # Number of hidden / recurrent units
    "HIDDEN_SIZE": 8,

    # CTRNN integration timestep (ignored for feedforward)
    "CTRNN_DT": 0.2,

    # ---- Genome encoding ----------------------------------------------------
    # 'binary':     Gray-coded bit strings, bit-flip mutation (default, original behaviour).
    # 'realvalued': Normalised floats in [0,1] with Gaussian mutation.
    #               MUTATION_RATE is reused as the Gaussian sigma (try 0.05).
    #               NEUTRALITY_P is ignored for real-valued encoding.
    # Switch here for Test 2 (real-valued encoding ablation).
    "GENOME_ENCODING": "binary",

    # ---- Bits per weight ----------------------------------------------------
    # Bits per weight/parameter (more bits = finer resolution + larger genome)
    # 16 bits gives ~0.0001 resolution over [-3, 3]
    # Only used when GENOME_ENCODING='binary'.
    "BITS_PER_WEIGHT": 16,

    # ---- Fitness weighting --------------------------------------------------
    # Must sum to 1.0
    "FITNESS_W_PERF": 0.45,    # weight on train performance
    "FITNESS_W_GEN":  0.55,    # weight on generalisation performance

    # ---- Neutrality ---------------------------------------------------------
    # Fraction of genome bits that are neutral padding (don't encode any weight).
    # p=0.0 -> no neutral bits (standard neuroevolution)
    # p=0.75 -> 75% of bits are neutral, matching NK landscape optimum
    # Neutral bits are appended after the active weight-encoding bits;
    # mutations there have zero effect on the network = neutral mutations.
    "NEUTRALITY_P": 0.75,

    # ---- Genetic Algorithm --------------------------------------------------
    "POPULATION_SIZE": 20,
    "GENERATIONS": 200,
    "MUTATION_RATE": 0.005,     # per-bit flip probability
    "TOURNAMENT_SIZE": 3,
    "RECORD_INTERVAL": 1,       # record stats every N generations
    "N_REPLICATES": 2,          # independent GA runs (different seeds)

    # ---- Evolution snapshots ------------------------------------------------
    # Generations at which to save the best-so-far genome for trajectory analysis.
    # Set to [] to disable snapshot saving.
    "SNAPSHOT_GENS": [0, 25, 50, 75, 100, 125, 150, 175, 200],

    # ---- Output -------------------------------------------------------------
    "OUTPUT_DIR": "/Users/jason1/Documents/Sussex_Uni/Adaptive_systems/Alife_Assignment/results_stage1",
    "VERBOSE": True,
}

# ============================================================
#  END OF HYPERPARAMETERS
# ============================================================


import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from genetic_algorithm import GeneticAlgorithm
from neuroevo_fitness import NeuroevoFitnessLandscape


def run_single_replicate(config, replicate_seed, replicate_idx):
    """
    Run one complete GA evolution with the given config.

    Args:
        config: hyperparameter dict
        replicate_seed: integer seed for this replicate
        replicate_idx: index label for logging

    Returns:
        dict of results
    """
    print(f"\n  [Replicate {replicate_idx + 1}] seed={replicate_seed}")

    # Build fitness landscape (also probes env dims + builds decoder)
    landscape = NeuroevoFitnessLandscape(config)

    print(f"  Genome length: {landscape.N} bits "
          f"(controller={config['CONTROLLER_TYPE']}, "
          f"hidden={config['HIDDEN_SIZE']}, "
          f"obs_dim={landscape.obs_dim}, act_dim={landscape.act_dim})")

    # Build GA — uses existing GA unchanged
    ga = GeneticAlgorithm(
        fitness_landscape=landscape,
        population_size=config['POPULATION_SIZE'],
        mutation_rate=config['MUTATION_RATE'],
        tournament_size=config['TOURNAMENT_SIZE'],
        seed=replicate_seed,
    )
    ga.initialize_population()

    # Track all-time best genome across all generations (not just final gen)
    best_genome = None
    best_fitness = -np.inf

    # Snapshot tracking: save best-so-far genome at specified generations
    n_gens = config['GENERATIONS']
    _default_snaps = sorted({1, max(2, n_gens // 5), max(3, n_gens // 2), n_gens})
    snapshot_gens_set = set(config.get('SNAPSHOT_GENS', _default_snaps))
    snapshots = []

    # Per-generation train/test split for the current-best individual
    split_history = {'best_train_score': [], 'best_test_score': []}

    t0 = time.time()
    ga.record_statistics()  # record gen 0
    # Log initial split (random population best)
    fitness_values = ga.get_population_fitness()
    init_best_idx = int(np.argmax(fitness_values))
    init_best = ga.population[init_best_idx]
    tr0, te0, _ = landscape.evaluate_split(init_best)
    split_history['best_train_score'].append(tr0)
    split_history['best_test_score'].append(te0)

    # Save gen 0 snapshot (initial random population best)
    if 0 in snapshot_gens_set:
        best_genome = init_best.copy().tolist()
        best_fitness = float(fitness_values[init_best_idx])
        snapshots.append({'gen': 0, 'genome': best_genome, 'fitness': best_fitness})

    for gen in range(config['GENERATIONS']):
        ga.evolve_generation()
        # Update all-time best
        fitness_values = ga.get_population_fitness()
        gen_best_idx = int(np.argmax(fitness_values))
        gen_best_fitness = float(fitness_values[gen_best_idx])
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = ga.population[gen_best_idx].copy().tolist()

        # Save snapshot of best-so-far genome at requested generations
        if (gen + 1) in snapshot_gens_set and best_genome is not None:
            snapshots.append({
                'gen': gen + 1,
                'genome': best_genome,   # already a list from .tolist()
                'fitness': best_fitness,
            })

        if (gen + 1) % config['RECORD_INTERVAL'] == 0:
            ga.record_statistics()
            # Log train/test split of current best individual (1 extra eval)
            tr, te, _ = landscape.evaluate_split(
                ga.population[int(np.argmax(fitness_values))])
            split_history['best_train_score'].append(tr)
            split_history['best_test_score'].append(te)
            if config['VERBOSE']:
                print(f"Gen {ga.generation}: "
                      f"Mean Fitness={np.mean(fitness_values):.3f}, "
                      f"Diversity={ga.history['hamming_distance'][-1]:.2f}, "
                      f"Train={tr:.3f}, Test={te:.3f}")
    history = ga.history
    elapsed = time.time() - t0

    final_stats = ga.get_summary_statistics()

    print(f"  Done in {elapsed:.1f}s | "
          f"Best fitness: {best_fitness:.4f} | "
          f"Mean fitness: {final_stats['mean_fitness']:.4f}")

    return {
        'replicate': replicate_idx,
        'seed': replicate_seed,
        'elapsed_seconds': elapsed,
        'final_stats': final_stats,
        'history': {
            **{k: [float(v) for v in vals] for k, vals in history.items()},
            **split_history,
        },
        'genome_length': landscape.N,
        'obs_dim': landscape.obs_dim,
        'act_dim': landscape.act_dim,
        'best_genome': best_genome,
        'best_fitness': best_fitness,
        'snapshots': snapshots,
    }


def run_stage1(config=None):
    """
    Run full Stage 1 experiment (N replicates).

    Args:
        config: optional override dict (defaults to module-level CONFIG)
    """
    if config is None:
        config = CONFIG

    output_path = Path(config['OUTPUT_DIR'])
    output_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Neuroevolution on "
          f"{config['ENV_TYPE'].upper()} "
          f"[{config['CONTROLLER_TYPE']}]")
    print(f"Population: {config['POPULATION_SIZE']} | "
          f"Generations: {config['GENERATIONS']} | "
          f"Replicates: {config['N_REPLICATES']}")
    print("=" * 60)

    all_results = []
    experiment_start = time.time()

    for rep_idx in range(config['N_REPLICATES']):
        rep_seed = rep_idx * 42 + 7   # deterministic but spread seeds
        result = run_single_replicate(config, rep_seed, rep_idx)
        all_results.append(result)

    total_time = time.time() - experiment_start

    # ---- Summary -----------------------------------------------------------
    max_fitnesses = [r['best_fitness'] for r in all_results]
    mean_fitnesses = [r['final_stats']['mean_fitness'] for r in all_results]

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Best fitness across replicates: {np.max(max_fitnesses):.4f}")
    print(f"Mean final fitness (+/- std):   "
          f"{np.mean(mean_fitnesses):.4f} +/- {np.std(mean_fitnesses):.4f}")
    print("=" * 60)

    # ---- Save results -------------------------------------------------------
    timestamp = int(time.time())
    output_file = (output_path /
                   f"stage1_{config['ENV_TYPE']}_{config['CONTROLLER_TYPE']}"
                   f"_{timestamp}.json")

    def _to_serializable(obj):
        """Recursively convert numpy types to Python native for JSON."""
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload = _to_serializable({
        'config': dict(config),
        'results': all_results,
        'summary': {
            'total_time_seconds': total_time,
            'best_fitness': float(np.max(max_fitnesses)),
            'mean_final_fitness': float(np.mean(mean_fitnesses)),
            'std_final_fitness': float(np.std(mean_fitnesses)),
        }
    })

    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to: {output_file}")
    return all_results, output_file


if __name__ == "__main__":
    run_stage1()
