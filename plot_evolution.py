"""
Evolution Analysis Plots
=========================
Generates figures showing how the CTRNN controller develops structured
behaviour over the course of evolution.

Five figures saved to results_stage1/:

  1. evolution_fitness_<stem>.png       — fitness curves (mean/best ± shading)
  2. evolution_trajectories_<stem>.png  — state proxy vs time for each snapshot
  3. evolution_phase_<stem>.png         — phase portrait (θ vs dθ) per snapshot
  4. evolution_actions_<stem>.png       — action vs time per snapshot
  5. evolution_overlay_<stem>.png       — final policy rolled out on multiple seeds

Usage:
    python plot_evolution.py path/to/results.json
    python plot_evolution.py          # auto-loads latest Acrobot file
"""
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent))
from genome_decoder import build_decoder_for_controller
from controllers import build_controller

# ---------------------------------------------------------------------------
#  Colour palette (consistent across all figures)
# ---------------------------------------------------------------------------
_SNAP_COLOURS = ['#c0392b', '#e67e22', '#8e44ad', '#27ae60', '#2980b9', '#16a085']  # early→final
_FIG_BG  = 'white'
_AXES_BG = '#f7f7f7'
_TEXT    = '#222222'
_GRID    = '#cccccc'

plt.rcParams.update({
    'figure.facecolor': _FIG_BG,
    'axes.facecolor':   _AXES_BG,
    'axes.edgecolor':   '#aaaaaa',
    'axes.labelcolor':  _TEXT,
    'xtick.color':      _TEXT,
    'ytick.color':      _TEXT,
    'text.color':       _TEXT,
    'grid.color':       _GRID,
    'grid.linewidth':   0.5,
    'savefig.facecolor': _FIG_BG,
})


# ---------------------------------------------------------------------------
#  Loader
# ---------------------------------------------------------------------------

def load_results(path=None, results_dir='results_stage1'):
    if path:
        with open(path) as f:
            return json.load(f), path
    files = sorted(glob.glob(f'{results_dir}/stage1_acrobot_*.json'))
    if not files:
        # Fall back to latest of any env
        files = sorted(glob.glob(f'{results_dir}/stage1_*.json'))
    if not files:
        raise FileNotFoundError(f"No results in {results_dir}/")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f), path


# ---------------------------------------------------------------------------
#  Controller builder
# ---------------------------------------------------------------------------

def decode_genome(genome_list, result, config):
    encoding = config.get('GENOME_ENCODING', 'binary')
    decoder = build_decoder_for_controller(
        controller_type=config.get('CONTROLLER_TYPE', 'ctrnn'),
        obs_dim=result['obs_dim'],
        act_dim=result['act_dim'],
        hidden_size=config['HIDDEN_SIZE'],
        bits_per_weight=config.get('BITS_PER_WEIGHT', 16),
    )
    if encoding == 'realvalued':
        genome = np.array(genome_list, dtype=np.float64)
        params = decoder.decode_realvalued(genome)
    else:
        genome = np.array(genome_list, dtype=int)
        params = decoder.decode(genome)
    ctrl = build_controller(
        config.get('CONTROLLER_TYPE', 'ctrnn'),
        result['obs_dim'], result['act_dim'],
        config['HIDDEN_SIZE'], dt=config.get('CTRNN_DT', 0.2)
    )
    ctrl.set_params(params)
    return ctrl


# ---------------------------------------------------------------------------
#  Environment rollout
# ---------------------------------------------------------------------------

def rollout(ctrl, env_type, max_steps, seed=0, env_params=None):
    """
    Roll out one episode.
    Returns obs_seq (list of obs arrays), action_seq (list), solved (bool).
    env_params: dict of attribute overrides applied to env.unwrapped (acrobot only).
    """
    if env_type == 'acrobot':
        env = gym.make('Acrobot-v1')
        if env_params:
            for k, v in env_params.items():
                setattr(env.unwrapped, k, v)
        continuous = False
    elif env_type == 'mountaincarcontinuous':
        env = gym.make('MountainCarContinuous-v0')
        continuous = True
    elif env_type == 'lunarlander':
        kwargs = dict(env_params) if env_params else {}
        env = gym.make('LunarLander-v3', **kwargs)
        continuous = False
    elif env_type == 'lunarlander_continuous':
        kwargs = dict(env_params) if env_params else {}
        env = gym.make('LunarLanderContinuous-v3', **kwargs)
        continuous = True
    else:
        raise ValueError(f"Unsupported env_type for plot_evolution: {env_type}")

    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    obs_seq = [obs.copy()]
    action_seq = []
    total_reward = 0.0

    for _ in range(max_steps):
        flat = np.asarray(obs, dtype=np.float32)
        logits = ctrl.forward(flat)
        if continuous:
            action = np.tanh(np.asarray(logits, dtype=np.float32))
            if env_type == 'lunarlander_continuous':
                # Store both channels: (main_engine, lateral)
                # main engine fires only when > 0; clip to [0,1] for display
                action_seq.append((float(np.clip(action[0], 0.0, 1.0)),
                                   float(action[1])))
            else:
                action_seq.append(float(action[0]))
        else:
            action = int(np.argmax(logits))
            action_seq.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        obs_seq.append(obs.copy())
        if terminated or truncated:
            break

    env.close()
    # For LunarLander: terminated fires on both crash (-100) and landing (+100).
    # Use total_reward > 0 to distinguish real landing from crash.
    if env_type in ('lunarlander', 'lunarlander_continuous'):
        solved = terminated and total_reward > 0
    else:
        solved = terminated  # terminated = goal reached; truncated = timeout
    return obs_seq, action_seq, solved


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------

def acrobot_tip_height(obs):
    """tip_height = -cos(θ1) - cos(θ1+θ2).  Goal ≥ 1.0"""
    c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
    return -c1 - (c1 * c2 - s1 * s2)

def acrobot_theta1(obs):
    return np.arctan2(obs[1], obs[0])

def acrobot_dtheta1(obs):
    return obs[4]

def acrobot_torque(action):
    """Discrete {0,1,2} → torque {-1,0,+1}"""
    return action - 1


def mcc_state(obs):
    """Returns (position, velocity)"""
    return obs[0], obs[1]


def lunar_vy(obs):
    """LunarLander vertical velocity (obs[3]). Negative = descending."""
    return float(obs[3])


# ---------------------------------------------------------------------------
#  Snapshot helpers
# ---------------------------------------------------------------------------

def snap_label(snap, config):
    gen = snap['gen']
    if gen == 0:
        return 'Gen 0 (initial)', 'initial'
    # Use the last snapshot gen as the reference for 'final'
    snap_gens = config.get('SNAPSHOT_GENS', [config['GENERATIONS']])
    last_snap = max(snap_gens) if snap_gens else config['GENERATIONS']
    if gen <= max(1, last_snap // 10):
        stage = 'early'
    elif gen < last_snap:
        stage = 'mid'
    else:
        stage = 'final'
    return f"Gen {gen} ({stage})", stage


_MAX_DISPLAY_SNAPS = 6   # max panels shown in trajectory / action figures


def prepare_snapshots(data, max_display=_MAX_DISPLAY_SNAPS):
    """
    Return the best replicate and its snapshots (sorted by gen).
    Falls back gracefully if no snapshots are saved.
    If there are more snapshots than max_display, selects a representative
    subset: always keeps the first and last, evenly spaces the rest.
    """
    results = data['results']
    best_rep = max(results, key=lambda r: r.get('best_fitness', 0))
    snaps = best_rep.get('snapshots', [])
    # Only append the final genome if no snapshots exist at all
    if not snaps:
        n = data['config']['GENERATIONS']
        snaps = [{
            'gen': n,
            'genome': best_rep['best_genome'],
            'fitness': best_rep['best_fitness'],
        }]
    snaps = sorted(snaps, key=lambda s: s['gen'])
    # Subsample to at most max_display panels, keeping first + last
    if max_display and len(snaps) > max_display:
        indices = [round(i * (len(snaps) - 1) / (max_display - 1))
                   for i in range(max_display)]
        snaps = [snaps[i] for i in sorted(set(indices))]
    return best_rep, snaps


# ---------------------------------------------------------------------------
#  Figure 1 — Fitness curves
# ---------------------------------------------------------------------------

# Fitness threshold at which the task is considered solved, per env type.
# Derived from each env's _score_episode() mapping:
#   Acrobot:              clip((tip_height + 2) / 4)  →  tip @ goal (1.0) = 0.75
#   MountainCarContinuous: clip((pos + 1.2) / 1.8)   →  car @ flag (0.45) ≈ 0.917
#   LunarLander:          clip((reward + 100) / 300)  →  landing bonus (100) = 0.667
_SOLVED_THRESHOLD = {
    'acrobot':               0.75,
    'mountaincarcontinuous': 0.917,
    'lunarlander':           0.667,
    'lunarlander_continuous': 0.667,
}

def fig_diversity(data, out_dir, stem):
    """Genetic diversity over generations.

    Single panel: mean pairwise Hamming distance (raw genome bits) — captures
    how genetically spread-out the population is.
    """
    config  = data['config']
    results = data['results']
    n_reps  = len(results)

    ri = config.get('RECORD_INTERVAL', 1)
    n_pts = len(results[0]['history']['hamming_distance'])
    gens  = np.arange(n_pts) * ri

    snap_gens = config.get('SNAPSHOT_GENS', [])
    plot_max_gen = max(snap_gens) if snap_gens else gens[-1]
    clip = np.searchsorted(gens, plot_max_gen, side='right')
    gens = gens[:clip]

    def stack(key):
        return np.array([r['history'][key][:clip] for r in results])

    hamming = stack('hamming_distance')           # mean pairwise Hamming (bits)
    pop_size = config.get('POPULATION_SIZE', 20)

    _rep_colours = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#f39c12']
    _rep_dashes  = [(4, 2), (2, 2), (6, 2, 1, 2), (4, 2, 4, 2), (1, 1), (6, 2)]

    enc_label = config.get('ENCODING', 'binary')
    p_label   = config.get('NEUTRALITY_P', 0)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle(
        f"Genetic Diversity — {config['ENV_TYPE'].upper()}  "
        f"[p={p_label}, enc={enc_label}, pop={pop_size}, gens={plot_max_gen}]",
        fontsize=12, fontweight='bold')

    # --- Hamming diversity ---
    for i, rep_h in enumerate(hamming):
        col  = _rep_colours[i % len(_rep_colours)]
        dash = _rep_dashes[i % len(_rep_dashes)]
        ax1.plot(gens, rep_h, color=col, lw=1.2, ls=(0, dash), alpha=0.75,
                 label=f'Rep {i+1}' if n_reps > 1 else None)
    if n_reps > 1:
        mean_h = hamming.mean(axis=0)
        ax1.plot(gens, mean_h, color='#2c3e50', lw=2.2,
                 label='Mean across replicates')
        ax1.fill_between(gens, hamming.min(axis=0), hamming.max(axis=0),
                         color='#2c3e50', alpha=0.12)

    genome_len = results[0].get('genome_length', None)
    if genome_len:
        max_possible = genome_len * 0.5
        ax1.axhline(max_possible, color='#95a5a6', lw=0.8, ls=':', alpha=0.8,
                    label=f'Max expected random ({max_possible:.0f} bits)')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean pairwise Hamming distance (bits)')
    ax1.legend(fontsize=9)
    ax1.grid(True)

    fig.tight_layout()
    out = out_dir / f'evolution_diversity_{stem}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def fig1_fitness_curves(data, out_dir, stem):
    config  = data['config']
    results = data['results']
    n_reps  = len(results)

    ri = config.get('RECORD_INTERVAL', 1)
    n_pts = len(results[0]['history']['max_fitness'])
    gens  = np.arange(n_pts) * ri

    # Clip to last snapshot gen if defined (nothing interesting beyond there)
    snap_gens = config.get('SNAPSHOT_GENS', [])
    plot_max_gen = max(snap_gens) if snap_gens else gens[-1]
    clip = np.searchsorted(gens, plot_max_gen, side='right')
    gens = gens[:clip]

    def stack(key):
        return np.array([r['history'][key][:clip] for r in results])

    max_f  = stack('max_fitness')
    mean_f = stack('mean_fitness')
    train_s = stack('best_train_score')
    test_s  = stack('best_test_score')

    # Best-so-far (running max) — this is what snapshot fitness values track
    best_so_far = np.maximum.accumulate(max_f, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Fitness Curves — {config['ENV_TYPE'].upper()}  "
        f"[p={config.get('NEUTRALITY_P', 0)}, pop={config['POPULATION_SIZE']}, "
        f"gens shown={plot_max_gen}/{config['GENERATIONS']}]",
        fontsize=12, fontweight='bold')

    # Top panel: global best across ALL replicates + individual rep lines for context
    global_best = best_so_far.max(axis=0)
    ax1.plot(gens, global_best, color='#2ecc71', lw=2.5, label='Best fitness (global across replicates)')
    _rep_colours = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#f39c12']
    _rep_dashes  = [(4, 2), (2, 2), (6, 2, 1, 2), (4, 2, 4, 2), (1, 1), (6, 2)]
    for i, rep_bsf in enumerate(best_so_far):
        col  = _rep_colours[i % len(_rep_colours)]
        dash = _rep_dashes[i % len(_rep_dashes)]
        ax1.plot(gens, rep_bsf, color=col, lw=1.2, ls=(0, dash), alpha=0.7,
                 label=f'Rep {i+1} best' if n_reps > 1 else None)

    # Mean fitness: global mean across all individuals in all replicates
    mean_avg = mean_f.mean(axis=0)
    ax1.plot(gens, mean_avg, color='#3498db', lw=2.2, label='Mean fitness ± range across replicates')
    ax1.fill_between(gens, mean_f.min(axis=0), mean_f.max(axis=0),
                     color='#3498db', alpha=0.15)

    ax1.axhline(1.0, color='#666666', lw=0.8, ls=':', alpha=0.6)

    # --- Solved threshold line ---
    env_type = config.get('ENV_TYPE', '').lower()
    solved_thresh = _SOLVED_THRESHOLD.get(env_type)
    if solved_thresh is not None:
        ax1.axhline(solved_thresh, color='#e67e22', lw=1.5, ls='--', alpha=0.9,
                    label=f'Task-solved threshold ({solved_thresh:.2f})')
        # Mark the first generation where any replicate's best-so-far crosses it
        crossed_gen = None
        for rep_bsf in best_so_far:
            idx = np.argmax(rep_bsf >= solved_thresh)
            if rep_bsf[idx] >= solved_thresh:
                g = int(gens[idx])
                if crossed_gen is None or g < crossed_gen:
                    crossed_gen = g
        if crossed_gen is not None:
            ax1.axvline(crossed_gen, color='#e67e22', lw=1.2, ls=':', alpha=0.7)
            ax1.annotate(
                f'First solved\n(gen {crossed_gen})',
                xy=(crossed_gen, solved_thresh),
                xytext=(crossed_gen + max(1, (gens[-1] - gens[0]) * 0.04), solved_thresh + 0.06),
                fontsize=7, color='#b7530a', alpha=0.95,
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=0.8),
            )

    ax1.set_ylabel('Fitness')
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # Bottom panel: train vs test score — mean line + shaded replicate range
    for arr, label, color in [
        (train_s, 'Train score (mean ± replicate range)', '#e67e22'),
        (test_s,  'Test score (mean ± replicate range)',  '#e74c3c'),
    ]:
        avg = arr.mean(axis=0)
        ax2.plot(gens, avg, color=color, lw=2.2, label=label)
        ax2.fill_between(gens, arr.min(axis=0), arr.max(axis=0),
                         color=color, alpha=0.15)

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True)

    fig.tight_layout()
    out = out_dir / f'evolution_fitness_{stem}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
#  Figure 2 — State trajectories
# ---------------------------------------------------------------------------

# Acrobot trajectory/action figures use LINK_MOI=2.0 (the highest-inertia test
# variant from ACROBOT_TEST_PARAMS). This is a real test condition — the display
# trajectory and the solved/failed label both refer to the same thing.
_ACROBOT_HARD_VARIANT = {'LINK_MOI': 2.0}


def fig2_state_trajectories(data, out_dir, stem):
    config   = data['config']
    env_type = config['ENV_TYPE']
    max_steps = config.get('ACROBOT_MAX_STEPS',
                config.get('MOUNTAINCARCONTINUOUS_MAX_STEPS',
                config.get('MAX_STEPS_PER_EPISODE', 500)))

    best_rep, snaps = prepare_snapshots(data)
    n_snaps = len(snaps)
    colours = _SNAP_COLOURS[:n_snaps]

    hard = _ACROBOT_HARD_VARIANT if env_type == 'acrobot' else None
    variant_note = '  [LINK_MOI=2.0 test variant]' if env_type == 'acrobot' else ''

    if env_type == 'acrobot':
        y_label = 'Tip height  (goal ≥ 1.0)'
        y_lim   = (-2.1, 2.1)
        goal_y  = 1.0
        def get_ys(obs_seq): return [acrobot_tip_height(o) for o in obs_seq]
    elif env_type in ('lunarlander', 'lunarlander_continuous'):
        y_label = 'Vertical velocity  vy  (m/s)'
        y_lim   = (-3.0, 3.0)
        goal_y  = None
        def get_ys(obs_seq): return [lunar_vy(o) for o in obs_seq]
    else:
        y_label = 'Velocity  (m/s)'
        y_lim   = (-0.08, 0.08)
        goal_y  = None
        def get_ys(obs_seq): return [mcc_state(o)[1] for o in obs_seq]

    # 3 rows × 2 cols grid — one panel per snapshot generation
    n_cols = 2
    n_rows = (n_snaps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.0 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    rep_idx = best_rep.get('replicate', 0) + 1
    solved_thresh = _SOLVED_THRESHOLD.get(env_type)
    fig.suptitle(
        f"State Trajectory by Generation — {env_type.upper()}{variant_note}  "
        f"[best replicate: rep {rep_idx}]",
        fontsize=12, fontweight='bold')

    for i, (snap, color) in enumerate(zip(snaps, colours)):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        ctrl = decode_genome(snap['genome'], best_rep, config)

        obs_seq, _, solved_raw = rollout(ctrl, env_type, max_steps, seed=0,
                                         env_params=hard)
        # Use snapshot fitness vs threshold — raw terminated flag is unreliable
        # on a single seed (good controllers often time-out rather than land).
        if solved_thresh is not None:
            snap_fit = snap.get('fitness', 0.0)
            passed = snap_fit >= solved_thresh
        else:
            passed = solved_raw
        status = '  (passed ✓)' if passed else '  (failed ✗)'
        label, _ = snap_label(snap, config)

        ys    = get_ys(obs_seq)
        steps = np.arange(len(ys))
        ax.plot(steps, ys, color=color, lw=1.8, zorder=3)
        if goal_y is not None:
            ax.axhline(goal_y, color='#2ecc71', lw=1.2, ls='--', alpha=0.8,
                       label='goal', zorder=2)
            ax.legend(fontsize=7)
        ax.set_xlabel('Time step')
        ax.set_ylabel(y_label)
        ax.set_ylim(*y_lim)
        ax.grid(True)
        ax.set_title(f"{label}\nfit={snap['fitness']:.4f}{status}",
                     fontsize=9, color=color)

    # Hide unused panels if n_snaps < n_rows * n_cols
    for j in range(n_snaps, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    out = out_dir / f'evolution_trajectories_{stem}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
#  Figure 3 — Phase portrait
# ---------------------------------------------------------------------------

def fig3_phase_portrait(data, out_dir, stem):
    config   = data['config']
    env_type = config['ENV_TYPE']
    max_steps = config.get('ACROBOT_MAX_STEPS',
                config.get('MOUNTAINCARCONTINUOUS_MAX_STEPS',
                config.get('MAX_STEPS_PER_EPISODE', 500)))

    best_rep, snaps = prepare_snapshots(data)
    n_snaps = len(snaps)
    colours = _SNAP_COLOURS[:n_snaps]

    if env_type == 'acrobot':
        x_label = 'θ₁  (joint 1 angle, rad)'
        y_label = 'dθ₁  (angular velocity, rad/s)'
    else:
        x_label = 'Position'
        y_label = 'Velocity'

    fig, axes = plt.subplots(1, n_snaps, figsize=(4.0 * n_snaps, 4.5), sharey=True)
    if n_snaps == 1:
        axes = [axes]
    fig.suptitle(
        f"Phase Portrait by Generation — {env_type.upper()}",
        fontsize=12, fontweight='bold')

    for ax, snap, color in zip(axes, snaps, colours):
        ctrl = decode_genome(snap['genome'], best_rep, config)
        obs_seq, _, _ = rollout(ctrl, env_type, max_steps, seed=0)
        steps_list = [len(rollout(ctrl, env_type, max_steps, seed=s)[0]) - 1
                      for s in range(10)
                      if rollout(ctrl, env_type, max_steps, seed=s)[2]]
        if steps_list:
            solve_str = f"solved {len(steps_list)}/10, mean {np.mean(steps_list):.0f} steps"
        else:
            solve_str = "0/10 solved"

        if env_type == 'acrobot':
            xs = [acrobot_theta1(o)  for o in obs_seq]
            ys = [acrobot_dtheta1(o) for o in obs_seq]
        else:
            xs = [mcc_state(o)[0] for o in obs_seq]
            ys = [mcc_state(o)[1] for o in obs_seq]

        # Colour by time (early=pale, late=saturated)
        n = len(xs)
        alphas = np.linspace(0.2, 1.0, n)
        for i in range(n - 1):
            ax.plot(xs[i:i+2], ys[i:i+2], color=color,
                    alpha=float(alphas[i]), lw=1.2)
        ax.plot(xs[0],  ys[0],  'o', ms=6,  color='white', zorder=5, label='start')
        ax.plot(xs[-1], ys[-1], '*', ms=10, color='gold',  zorder=5, label='end')

        label, _ = snap_label(snap, config)
        ax.set_title(f"{label}\n{solve_str}", fontsize=9, color=color)
        ax.set_xlabel(x_label)
        ax.grid(True)
        if ax is axes[0]:
            ax.set_ylabel(y_label)
            ax.legend(fontsize=8, framealpha=0.4)

    fig.tight_layout()
    out = out_dir / f'evolution_phase_{stem}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
#  Figure 4 — Action sequences
# ---------------------------------------------------------------------------

def fig4_action_sequences(data, out_dir, stem):
    config   = data['config']
    env_type = config['ENV_TYPE']
    max_steps = config.get('ACROBOT_MAX_STEPS',
                config.get('MOUNTAINCARCONTINUOUS_MAX_STEPS',
                config.get('MAX_STEPS_PER_EPISODE', 500)))

    best_rep, snaps = prepare_snapshots(data)
    n_snaps = len(snaps)
    colours = _SNAP_COLOURS[:n_snaps]

    continuous = (env_type in ('mountaincarcontinuous', 'lunarlander_continuous'))
    lunar_cont = (env_type == 'lunarlander_continuous')
    if lunar_cont:
        a_label = 'Main engine thrust  [0=off, 1=full]'
    elif continuous:
        a_label = 'Thrust force'
    elif env_type == 'lunarlander':
        a_label = 'Action  {0=none,1=left,2=main,3=right}'
    else:
        a_label = 'Torque  {-1, 0, +1}'
    hard = _ACROBOT_HARD_VARIANT if env_type == 'acrobot' else None
    variant_note = '  [LINK_MOI=2.0 test variant]' if env_type == 'acrobot' else ''

    n_cols = min(2, n_snaps)
    n_rows = -(-n_snaps // n_cols)  # ceiling division
    fig, axes_grid = plt.subplots(n_rows, n_cols,
                                  figsize=(4.5 * n_cols, 3.5 * n_rows),
                                  sharey=True)
    axes_flat = np.array(axes_grid).flatten()
    # hide any unused panels in the last row
    for _ax in axes_flat[n_snaps:]:
        _ax.set_visible(False)
    rep_idx = best_rep.get('replicate', 0) + 1
    fig.suptitle(
        f"Action Sequences by Generation — {env_type.upper()}{variant_note}  "
        f"[best replicate: rep {rep_idx}]",
        fontsize=12, fontweight='bold')

    solved_thresh = _SOLVED_THRESHOLD.get(env_type)

    for i, (ax, snap, color) in enumerate(zip(axes_flat, snaps, colours)):
        ctrl = decode_genome(snap['genome'], best_rep, config)
        _, action_seq, solved_hard = rollout(ctrl, env_type, max_steps, seed=0,
                                              env_params=hard)
        if env_type == 'acrobot':
            solve_str = 'passed ✓' if solved_hard else 'failed ✗'
        elif solved_thresh is not None:
            # Use snapshot fitness vs solved threshold — avoids single-seed
            # stochasticity (a good controller can time-out on one bad seed
            # without triggering 'terminated', making solved=False despite
            # high fitness).
            snap_fit = snap.get('fitness', 0.0)
            passed = snap_fit >= solved_thresh
            solve_str = f"{'passed ✓' if passed else 'failed ✗'}  (f={snap_fit:.3f})"
        else:
            _, _, solved = rollout(ctrl, env_type, max_steps, seed=0)
            solve_str = 'passed ✓' if solved else 'failed ✗'

        if lunar_cont:
            main_seq   = [a[0] for a in action_seq]   # clipped to [0,1]
            lateral_seq = [a[1] for a in action_seq]  # raw [-1,1]
        elif not continuous:
            if env_type == 'lunarlander':
                torques = [int(a) for a in action_seq]
            else:
                torques = [acrobot_torque(a) for a in action_seq]
        else:
            torques = [float(a) for a in action_seq]

        steps = np.arange(len(action_seq))
        if lunar_cont:
            ax.step(steps, main_seq,    where='post', color=color,     lw=1.5, label='main (fires >0)')
            ax.step(steps, lateral_seq, where='post', color='#888888', lw=1.2,
                    ls='--', label='lateral (±0.5 deadband)')
            ax.fill_between(steps, 0, main_seq, step='post', color=color, alpha=0.18)
            ax.axhline(0,    color='#aaaaaa', lw=0.6, ls=':')
            ax.axhline(0.5,  color='#888888', lw=0.5, ls=':', alpha=0.5)
            ax.axhline(-0.5, color='#888888', lw=0.5, ls=':', alpha=0.5)
            if ax is axes_flat[0]:
                ax.legend(fontsize=7, loc='upper right')
        else:
            ax.step(steps, torques, where='post', color=color, lw=1.5)
            ax.fill_between(steps, 0, torques, step='post', color=color, alpha=0.25)
            ax.axhline(0, color='#aaaaaa', lw=0.6, ls=':')

        label, _ = snap_label(snap, config)
        ax.set_title(f"{label}\n{solve_str}", fontsize=9, color=color)
        ax.set_xlabel('Time step')
        ax.grid(True, axis='y')
        if i % n_cols == 0:
            ax.set_ylabel(a_label)
        if lunar_cont:
            ax.set_ylim(-1.1, 1.1)
        elif not continuous:
            if env_type == 'lunarlander':
                ax.set_yticks([0, 1, 2, 3])
            else:
                ax.set_yticks([-1, 0, 1])
        else:
            ax.set_ylim(-1.1, 1.1)


    fig.tight_layout()
    out = out_dir / f'evolution_actions_{stem}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
#  Figure 5 — Episode overlay (final policy, multiple seeds)
# ---------------------------------------------------------------------------

def fig5_episode_overlay(data, out_dir, stem, n_seeds=12):
    config   = data['config']
    env_type = config['ENV_TYPE']
    max_steps = config.get('ACROBOT_MAX_STEPS',
                config.get('MOUNTAINCARCONTINUOUS_MAX_STEPS',
                config.get('MAX_STEPS_PER_EPISODE', 500)))

    results = data['results']

    if env_type == 'acrobot':
        goal_y  = 1.0
        y_label = 'Tip height'
        y_lim   = (-2.1, 2.1)
    elif env_type in ('lunarlander', 'lunarlander_continuous'):
        goal_y  = None
        y_label = 'Vertical velocity vy'
        y_lim   = (-3.0, 3.0)
    else:
        goal_y  = 0.45
        y_label = 'Position'
        y_lim   = (-1.3, 0.7)

    fig, axes = plt.subplots(1, len(results),
                             figsize=(6 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    fig.suptitle(
        f"Final Policy — Episode Overlay ({n_seeds} seeds) — {env_type.upper()}",
        fontsize=12, fontweight='bold')

    for ax, result in zip(axes, results):
        ctrl = decode_genome(result['best_genome'], result, config)
        n_solved = 0
        for seed in range(n_seeds):
            obs_seq, _, solved = rollout(ctrl, env_type, max_steps, seed=seed)
            if env_type == 'acrobot':
                ys = [acrobot_tip_height(o) for o in obs_seq]
            elif env_type in ('lunarlander', 'lunarlander_continuous'):
                ys = [lunar_vy(o) for o in obs_seq]
            else:
                ys = [mcc_state(o)[0] for o in obs_seq]
            color   = '#2ecc71' if solved else '#e74c3c'
            alpha   = 0.65 if solved else 0.35
            lw      = 1.6 if solved else 0.8
            ax.plot(np.arange(len(ys)), ys, color=color, alpha=alpha, lw=lw)
            if solved:
                n_solved += 1

        ax.axhline(goal_y, color='white', lw=1.2, ls='--', alpha=0.7)
        ax.set_ylim(*y_lim)
        ax.set_xlabel('Time step')
        ax.set_ylabel(y_label)
        ax.set_title(
            f"Rep {result['replicate']+1} | fit={result['best_fitness']:.4f} | "
            f"solved {n_solved}/{n_seeds} seeds",
            fontsize=9)
        ax.grid(True)

        # Legend proxies
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color='#2ecc71', lw=1.6, label='solved'),
            Line2D([0], [0], color='#e74c3c', lw=0.8, alpha=0.5, label='not solved'),
        ], fontsize=8, framealpha=0.4)

    fig.tight_layout()
    out = out_dir / f'evolution_overlay_{stem}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data, results_path = load_results(path_arg)

    config   = data['config']
    env_type = config['ENV_TYPE']
    stem     = Path(results_path).stem
    out_dir  = Path(results_path).parent

    print(f"Env: {env_type.upper()}  |  "
          f"Replicates: {len(data['results'])}  |  "
          f"Generations: {config['GENERATIONS']}")

    snaps_in_data = any(r.get('snapshots') for r in data['results'])
    if not snaps_in_data:
        print("WARNING: No snapshots found in results file. "
              "Figures 2-4 will use only the final best genome. "
              "Re-run with SNAPSHOT_GENS set to get per-generation views.")

    fig_diversity(data, out_dir, stem)
    fig1_fitness_curves(data, out_dir, stem)
    fig2_state_trajectories(data, out_dir, stem)

    if env_type == 'mountaincarcontinuous':
        # Phase portrait is the key figure for MCC — shows cyclic momentum buildup
        fig3_phase_portrait(data, out_dir, stem)
        fig4_action_sequences(data, out_dir, stem)
        fig5_episode_overlay(data, out_dir, stem)
    else:
        # Acrobot: action plot is optional but included; phase/overlay skipped
        fig4_action_sequences(data, out_dir, stem)

    print("\nAll figures saved.")


if __name__ == '__main__':
    main()
