"""
Stage 1 Results Plotter
========================
Generates a 2-panel figure per results file:
  Top:    Fitness curves over generations (mean fitness, best-in-gen,
          best-genome train score, best-genome test score) — one line per replicate.
  Bottom: Per-variant bar chart showing train vs test scores for the best genome,
          re-evaluated with the correct scoring.

Usage:
    python plot_results.py                  # auto-loads latest JSON
    python plot_results.py path/to/file.json
"""
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Make sure imports resolve regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from genome_decoder import build_decoder_for_controller
from controllers import build_controller
from neuroevo_fitness import (
    _run_episode,
    _cartpole_flat, _acrobot_flat, _lunarlander_flat,
    _mountaincar_flat, _minigrid_flat,
    _make_cartpole, _make_acrobot, _make_lunarlander, _make_mountaincar,
    _make_mountaincarcontinuous,
    _acrobot_progress, _mountaincar_progress, _lunarlander_progress,
)
try:
    import minigrid  # noqa
except ModuleNotFoundError:
    pass
import gymnasium as gym

# ---------------------------------------------------------------------------
_OBS_FN = {
    'cartpole':              _cartpole_flat,
    'acrobot':               _acrobot_flat,
    'lunarlander':           _lunarlander_flat,
    'mountaincar':           _mountaincar_flat,
    'mountaincarcontinuous': _mountaincar_flat,
    'minigrid':              _minigrid_flat,
}
_ENV_FACTORY = {
    'cartpole':              _make_cartpole,
    'acrobot':               _make_acrobot,
    'lunarlander':           _make_lunarlander,
    'mountaincar':           _make_mountaincar,
    'mountaincarcontinuous': _make_mountaincarcontinuous,
}

RESULTS_DIR = Path(__file__).parent / 'results_stage1'


def load_latest(path=None):
    if path:
        return json.load(open(path)), Path(path)
    files = list(RESULTS_DIR.glob('*.json'))
    if not files:
        raise FileNotFoundError(f"No JSON files in {RESULTS_DIR}")
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading: {latest}")
    return json.load(open(latest)), latest


def score_episode(env_type, reward, success, max_progress, max_steps):
    if env_type == 'minigrid':
        return float(success)
    elif env_type == 'acrobot':
        return float(np.clip((max_progress + 2.0) / 4.0, 0, 1))
    elif env_type in ('mountaincar', 'mountaincarcontinuous'):
        return float(np.clip((max_progress + 1.2) / 1.8, 0, 1))
    elif env_type == 'lunarlander':
        return float(np.clip((max_progress + 3.0) / 5.0, 0.0, 1.0))
    else:
        return float(np.clip(reward, 0, max_steps) / max_steps)


def eval_variant(spec, env_type, controller, seeds, n_eps, max_steps):
    obs_fn   = _OBS_FN[env_type]
    factory  = _ENV_FACTORY.get(env_type)
    progress_fn = {'acrobot': _acrobot_progress,
                   'mountaincar': _mountaincar_progress,
                   'mountaincarcontinuous': _mountaincar_progress,
                   'lunarlander': _lunarlander_progress}.get(env_type)
    if isinstance(spec, dict):
        env = factory(spec) if factory else gym.make(spec)
    else:
        env = gym.make(spec)
    ep_scores = []
    for seed in seeds:
        for ep_i in range(n_eps):
            ep_seed = seed if ep_i == 0 else None
            reward, success, max_progress = _run_episode(
                env, controller, max_steps, obs_fn,
                seed=ep_seed, progress_fn=progress_fn,
                continuous_action=(env_type == 'mountaincarcontinuous'))
            ep_scores.append(score_episode(env_type, reward, success, max_progress, max_steps))
    env.close()
    return float(np.mean(ep_scores))


def variant_label(spec, idx):
    if isinstance(spec, dict):
        if not spec:
            return 'standard'
        return ', '.join(f'{k}={v}' for k, v in spec.items())
    return str(spec)


def run(path=None):
    d, fpath = load_latest(path)
    config   = d['config']
    env_type = config.get('ENV_TYPE', 'cartpole')
    n_reps   = len(d['results'])

    max_steps = config.get(
        'ACROBOT_MAX_STEPS' if env_type == 'acrobot'
        else 'MOUNTAINCAR_MAX_STEPS' if env_type == 'mountaincar'
        else 'MOUNTAINCARCONTINUOUS_MAX_STEPS' if env_type == 'mountaincarcontinuous'
        else 'MAX_STEPS_PER_EPISODE', 500)

    _PARAM_KEYS = {
        'cartpole':              ('CARTPOLE_TRAIN_PARAMS',              'CARTPOLE_TEST_PARAMS'),
        'acrobot':               ('ACROBOT_TRAIN_PARAMS',               'ACROBOT_TEST_PARAMS'),
        'lunarlander':           ('LUNARLANDER_TRAIN_PARAMS',           'LUNARLANDER_TEST_PARAMS'),
        'mountaincar':           ('MOUNTAINCAR_TRAIN_PARAMS',           'MOUNTAINCAR_TEST_PARAMS'),
        'mountaincarcontinuous': ('MOUNTAINCARCONTINUOUS_TRAIN_PARAMS', 'MOUNTAINCARCONTINUOUS_TEST_PARAMS'),
        'minigrid':              ('MINIGRID_TRAIN_ENVS',                'MINIGRID_TEST_ENVS'),
    }
    train_key, test_key = _PARAM_KEYS[env_type]
    train_specs = config.get(train_key, [{}])
    test_specs  = config.get(test_key,  [{}])
    n_eps       = config.get('N_EVAL_EPISODES', 2)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Stage 1 Neuroevolution — {env_type.upper()}  "
        f"[{config.get('CONTROLLER_TYPE','ctrnn').upper()}, "
        f"p_neutral={config.get('NEUTRALITY_P',0)}, "
        f"pop={config.get('POPULATION_SIZE')}, "
        f"gens={config.get('GENERATIONS')}]",
        fontsize=13, fontweight='bold')

    # ---- Top panel: fitness curves ----------------------------------------
    ax1 = fig.add_subplot(2, 1, 1)
    for i, r in enumerate(d['results']):
        h   = r['history']
        gen = h['generation']
        c   = colors[i % len(colors)]
        ax1.plot(gen, h['mean_fitness'],    color=c, lw=1.5, alpha=0.6,
                 linestyle='--', label=f'Rep {i+1} mean fitness')
        ax1.plot(gen, h['max_fitness'],     color=c, lw=2.0,
                 label=f'Rep {i+1} best fitness')
        if h.get('best_train_score'):
            ax1.plot(gen, h['best_train_score'], color=c, lw=1.2, alpha=0.8,
                     linestyle=':', label=f'Rep {i+1} train score')
        if h.get('best_test_score'):
            ax1.plot(gen, h['best_test_score'],  color=c, lw=1.2, alpha=0.8,
                     linestyle='-.', label=f'Rep {i+1} test score')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score (0–1)')
    ax1.set_title('Evolution of fitness over generations')
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # ---- Bottom panel: per-variant bar chart --------------------------------
    ax2 = fig.add_subplot(2, 1, 2)
    all_labels = (
        [('TRAIN: ' + variant_label(s, i)) for i, s in enumerate(train_specs)] +
        [('TEST: '  + variant_label(s, i)) for i, s in enumerate(test_specs)]
    )
    x = np.arange(len(all_labels))
    width = 0.8 / max(n_reps, 1)
    offsets = np.linspace(-(n_reps - 1) * width / 2,
                           (n_reps - 1) * width / 2, n_reps)

    print("\nPer-variant re-evaluation:")
    for i, r in enumerate(d['results']):
        genome = np.array(r['best_genome'], dtype=int)
        decoder = build_decoder_for_controller(
            'ctrnn', r['obs_dim'], r['act_dim'],
            config['HIDDEN_SIZE'], config['BITS_PER_WEIGHT'])
        ctrl = build_controller(
            'ctrnn', r['obs_dim'], r['act_dim'],
            config['HIDDEN_SIZE'], config.get('CTRNN_DT', 0.2))
        ctrl.set_params(decoder.decode(genome))

        scores = []
        for spec in train_specs:
            s = eval_variant(spec, env_type, ctrl,
                             config['TRAIN_SEEDS'], n_eps, max_steps)
            scores.append(s)
        for spec in test_specs:
            s = eval_variant(spec, env_type, ctrl,
                             config['TEST_SEEDS'], n_eps, max_steps)
            scores.append(s)

        n_train = len(train_specs)
        train_mean = np.mean(scores[:n_train])
        test_mean  = np.mean(scores[n_train:])
        print(f"  Rep {i+1}: train={train_mean:.3f}  test={test_mean:.3f}")
        for lbl, sc in zip(all_labels, scores):
            print(f"    {lbl:50s}  {sc:.3f}")

        bar_colors = ['#4c9ed9'] * n_train + ['#e07b54'] * len(test_specs)
        ax2.bar(x + offsets[i], scores, width,
                label=f'Rep {i+1} (best={r["best_fitness"]:.3f})',
                color=bar_colors, alpha=0.85, edgecolor='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Score (0–1)')
    ax2.set_title('Per-variant performance (blue = train, orange = test/held-out)')
    ax2.set_ylim(0, 1.05)
    ax2.axvline(len(train_specs) - 0.5, color='black', lw=1.5, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = fpath.parent / (fpath.stem + '_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(path)
