"""
Acrobot CTRNN Animation
========================
Loads the best genome from each replicate in the latest acrobot results file,
rolls out an episode with the CTRNN controller, and saves an animated GIF
showing both replicates side-by-side.

Acrobot geometry:
  Joint 1 at origin (0, 0).
  Link 1: length L1, angle theta1 from vertical (downward = 0).
  Link 2: length L2, angle theta2 relative to link 1.

Obs: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]
"""
import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from genome_decoder import build_decoder_for_controller
from controllers import build_controller
import gymnasium as gym


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(path=None, results_dir='results_stage1'):
    if path:
        with open(path) as f:
            return json.load(f), path
    files = sorted(glob.glob(f'{results_dir}/stage1_acrobot_*.json'))
    if not files:
        raise FileNotFoundError(f"No acrobot results in {results_dir}/")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f), path


def decode_controller(result, config):
    genome = np.array(result['best_genome'], dtype=int)
    decoder = build_decoder_for_controller(
        controller_type='ctrnn',
        obs_dim=result['obs_dim'],
        act_dim=result['act_dim'],
        hidden_size=config['HIDDEN_SIZE'],
        bits_per_weight=config['BITS_PER_WEIGHT'],
    )
    params = decoder.decode(genome)
    ctrl = build_controller(
        'ctrnn', result['obs_dim'], result['act_dim'],
        config['HIDDEN_SIZE'], dt=config.get('CTRNN_DT', 0.2)
    )
    ctrl.set_params(params)
    return ctrl


# ---------------------------------------------------------------------------
# Roll out one episode, collecting frames (raw obs each step)
# ---------------------------------------------------------------------------

def rollout(ctrl, max_steps=500, seed=0):
    env = gym.make('Acrobot-v1')
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    frames = [obs.copy()]
    total_reward = 0.0
    solved_at = None

    for step in range(max_steps):
        flat = np.asarray(obs, dtype=np.float32)
        logits = ctrl.forward(flat)
        action = int(np.argmax(logits))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(obs.copy())
        if terminated:
            solved_at = step + 1
            break
        if truncated:
            break

    env.close()
    return frames, total_reward, solved_at


# ---------------------------------------------------------------------------
# Draw one Acrobot frame onto an axes
# ---------------------------------------------------------------------------

def acrobot_xy(obs, L1=1.0, L2=1.0):
    """
    Compute joint and tip positions from obs.
    Returns: (jx0, jy0), (jx1, jy1), (tip_x, tip_y)
    """
    c1, s1, c2, s2 = obs[0], obs[1], obs[2], obs[3]
    # theta1: angle of link 1 from downward vertical
    # joint 1 at origin
    jx0, jy0 = 0.0, 0.0
    # end of link 1
    jx1 = L1 * s1          # x = L sin(theta1)
    jy1 = -L1 * c1         # y = -L cos(theta1)  (down = negative y)
    # theta1+theta2 via angle addition
    c12 = c1 * c2 - s1 * s2
    s12 = s1 * c2 + c1 * s2
    tip_x = jx1 + L2 * s12
    tip_y = jy1 - L2 * c12
    return (jx0, jy0), (jx1, jy1), (tip_x, tip_y)


def draw_acrobot(ax, obs, color1='steelblue', color2='tomato', alpha=1.0):
    (jx0, jy0), (jx1, jy1), (tip_x, tip_y) = acrobot_xy(obs)
    # Goal line: tip must be above y = 1.0 (Acrobot-v1 threshold)
    ax.axhline(1.0, color='green', linewidth=1.2, linestyle='--', alpha=0.6, zorder=1)
    # Links
    ax.plot([jx0, jx1], [jy0, jy1], lw=6, color=color1, alpha=alpha,
            solid_capstyle='round', zorder=3)
    ax.plot([jx1, tip_x], [jy1, tip_y], lw=6, color=color2, alpha=alpha,
            solid_capstyle='round', zorder=3)
    # Joints
    ax.plot(jx0, jy0, 'o', ms=10, color='k', zorder=4)
    ax.plot(jx1, jy1, 'o', ms=8,  color='k', zorder=4)
    ax.plot(tip_x, tip_y, 'o', ms=6, color='gold', zorder=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data, results_path = load_results(path_arg)
    config = data['config']
    results = data['results']
    n_rep = len(results)

    MAX_STEPS = config.get('ACROBOT_MAX_STEPS', 500)
    SEED = 0

    # Roll out each replicate
    all_frames = []
    infos = []
    for i, result in enumerate(results):
        if 'best_genome' not in result:
            print(f"Rep {i+1}: no best_genome — skipping")
            continue
        ctrl = decode_controller(result, config)
        frames, total_reward, solved_at = rollout(ctrl, max_steps=MAX_STEPS, seed=SEED)
        all_frames.append(frames)
        fitness = result.get('best_fitness', '?')
        label = (f"Rep {i+1} | seed={result['seed']} | "
                 f"fit={fitness:.4f} | steps={len(frames)-1}"
                 + (f" (SOLVED at {solved_at})" if solved_at else " (not solved)"))
        infos.append(label)
        print(label)

    if not all_frames:
        print("No replicates to animate.")
        return

    # Pad shorter rollouts with their last frame so all same length
    max_len = max(len(f) for f in all_frames)
    for fl in all_frames:
        while len(fl) < max_len:
            fl.append(fl[-1])

    # Build figure: one axes per replicate, side by side
    fig, axes = plt.subplots(1, len(all_frames), figsize=(5 * len(all_frames), 5))
    if len(all_frames) == 1:
        axes = [axes]

    fig.patch.set_facecolor('#1a1a2e')
    lim = 2.3
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    legend_patches = [
        mpatches.Patch(color='steelblue', label='Link 1'),
        mpatches.Patch(color='tomato',    label='Link 2'),
        mpatches.Patch(color='green',     label='Goal height', linestyle='--'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               facecolor='#1a1a2e', labelcolor='white', fontsize=9,
               framealpha=0.5)

    title_obj = fig.suptitle('', fontsize=10, color='white', y=1.01)

    def init():
        for ax in axes:
            ax.cla()
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_facecolor('#1a1a2e')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
        return []

    def update(frame_idx):
        init()
        for ax, frames, info in zip(axes, all_frames, infos):
            obs = frames[frame_idx]
            draw_acrobot(ax, obs)
            ax.set_title(info, color='white', fontsize=7.5, pad=4)
            ax.tick_params(colors='#888')
            # Tip height indicator
            _, _, (_, tip_y) = acrobot_xy(obs)
            ax.set_xlabel(f'step {frame_idx}  |  tip_y={tip_y:.2f}',
                          color='#aaa', fontsize=8)
        title_obj.set_text(
            f"Acrobot CTRNN — p={config.get('NEUTRALITY_P', 0)} "
            f"hidden={config['HIDDEN_SIZE']}")
        return []

    ani = FuncAnimation(fig, update, frames=max_len, init_func=init,
                        interval=40, blit=False)

    out_path = Path(results_path).parent / \
        f"acrobot_animation_{Path(results_path).stem}.gif"
    print(f"Saving {max_len} frames to {out_path} ...")
    ani.save(str(out_path), writer=PillowWriter(fps=25))
    print(f"Done → {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
