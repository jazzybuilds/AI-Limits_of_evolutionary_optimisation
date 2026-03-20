"""Re-evaluate best genome from each replicate, splitting train vs test score."""
import json, sys, numpy as np
sys.path.insert(0, '.')
import minigrid  # noqa
import gymnasium as gym
from genome_decoder import build_decoder_for_controller
from controllers import build_controller
from neuroevo_fitness import (
    _run_episode,
    _minigrid_flat, _cartpole_flat, _acrobot_flat,
    _lunarlander_flat, _mountaincar_flat,
    _make_cartpole, _make_acrobot, _make_lunarlander, _make_mountaincar,
    _acrobot_progress, _mountaincar_progress, _lunarlander_progress,
    NeuroevoFitnessLandscape,
)

_OBS_FN = {
    'cartpole':    _cartpole_flat,
    'acrobot':     _acrobot_flat,
    'lunarlander': _lunarlander_flat,
    'mountaincar': _mountaincar_flat,
    'minigrid':    _minigrid_flat,
}
_ENV_FACTORY = {
    'cartpole':    _make_cartpole,
    'acrobot':     _make_acrobot,
    'lunarlander': _make_lunarlander,
    'mountaincar': _make_mountaincar,
}

RESULTS_FILE = None  # set to a specific path, or None to use latest (sorted by filename)

import glob
from pathlib import Path

def load_results(path=None):
    if path is None:
        files = sorted(glob.glob('results_stage1/*.json'))
        # prefer the latest by modification time
        path = max(files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading: {path}")
    return json.load(open(path)), path


def eval_envs(env_specs, seeds, controller, env_type, n_eps=2, max_steps=500):
    """Evaluate controller across env_specs, returning (label, score) pairs."""
    obs_fn      = _OBS_FN.get(env_type, _cartpole_flat)
    env_factory = _ENV_FACTORY.get(env_type)
    scores = []
    for spec in env_specs:
        if isinstance(spec, dict):
            env   = env_factory(spec)
            label = 'standard' if not spec else str(spec)
        else:
            env   = gym.make(spec)
            label = spec
        ep_scores = []
        for seed in seeds:
            for ep_i in range(n_eps):
                ep_seed = seed if ep_i == 0 else None
                progress_fn = None
                if env_type == 'acrobot':
                    progress_fn = _acrobot_progress
                elif env_type == 'mountaincar':
                    progress_fn = _mountaincar_progress
                elif env_type == 'lunarlander':
                    progress_fn = _lunarlander_progress
                reward, success, max_progress = _run_episode(
                    env, controller, max_steps, obs_fn,
                    seed=ep_seed, progress_fn=progress_fn)
                if env_type == 'minigrid':
                    ep_scores.append(float(success))
                elif env_type == 'acrobot':
                    ep_scores.append(float(np.clip((max_progress + 2.0) / 4.0, 0, 1)))
                elif env_type == 'mountaincar':
                    ep_scores.append(float(np.clip((max_progress + 1.2) / 1.8, 0, 1)))
                elif env_type == 'lunarlander':
                    ep_scores.append(float(np.clip((max_progress + 3.0) / 5.0, 0.0, 1.0)))
                else:  # cartpole
                    ep_scores.append(
                        float(np.clip(reward, 0, max_steps) / max_steps))
        env.close()
        scores.append((label, float(np.mean(ep_scores))))
    return scores


_PARAM_KEYS = {
    'cartpole':    ('CARTPOLE_TRAIN_PARAMS',    'CARTPOLE_TEST_PARAMS'),
    'acrobot':     ('ACROBOT_TRAIN_PARAMS',     'ACROBOT_TEST_PARAMS'),
    'lunarlander': ('LUNARLANDER_TRAIN_PARAMS', 'LUNARLANDER_TEST_PARAMS'),
    'mountaincar': ('MOUNTAINCAR_TRAIN_PARAMS', 'MOUNTAINCAR_TEST_PARAMS'),
    'minigrid':    ('MINIGRID_TRAIN_ENVS',      'MINIGRID_TEST_ENVS'),
}

d, results_path = load_results(RESULTS_FILE)
config    = d['config']
env_type  = config.get('ENV_TYPE', 'cartpole')
train_key, test_key = _PARAM_KEYS.get(env_type, ('CARTPOLE_TRAIN_PARAMS', 'CARTPOLE_TEST_PARAMS'))
train_specs = config.get(train_key, [{}])
test_specs  = config.get(test_key,  [{}])
n_eps       = config.get('N_EVAL_EPISODES', 2)
max_steps   = config.get('ACROBOT_MAX_STEPS' if env_type == 'acrobot'
                         else 'MOUNTAINCAR_MAX_STEPS' if env_type == 'mountaincar'
                         else 'MAX_STEPS_PER_EPISODE', 500)

for i, r in enumerate(d['results']):
    genome = np.array(r['best_genome'], dtype=int)
    obs_dim, act_dim = r['obs_dim'], r['act_dim']

    decoder = build_decoder_for_controller(
        'ctrnn', obs_dim, act_dim,
        config['HIDDEN_SIZE'], config['BITS_PER_WEIGHT'])
    ctrl = build_controller(
        'ctrnn', obs_dim, act_dim,
        config['HIDDEN_SIZE'], config['CTRNN_DT'])
    ctrl.set_params(decoder.decode(genome))

    train_scores = eval_envs(train_specs, config['TRAIN_SEEDS'], ctrl, env_type, n_eps, max_steps)
    test_scores  = eval_envs(test_specs,  config['TEST_SEEDS'],  ctrl, env_type, n_eps, max_steps)

    train_mean = np.mean([s for _, s in train_scores])
    test_mean  = np.mean([s for _, s in test_scores])
    composite  = 0.45 * train_mean + 0.55 * test_mean

    print(f"\nReplicate {i+1} (seed={r['seed']}, stored best={r['best_fitness']:.4f}):")
    print(f"  TRAIN environments:")
    for label, score in train_scores:
        print(f"    {label:45s}  {score:.3f}")
    print(f"  Mean train: {train_mean:.3f}")
    print(f"  TEST environments (held-out):")
    for label, score in test_scores:
        print(f"    {label:45s}  {score:.3f}")
    print(f"  Mean test:  {test_mean:.3f}")
    print(f"  Composite fitness: {composite:.3f}")
    if train_mean > 0:
        print(f"  Generalisation ratio (test/train): {test_mean/train_mean:.2f}")

