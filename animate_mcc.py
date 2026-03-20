"""
MountainCarContinuous CTRNN Animation
======================================
Loads the best genome from each replicate in the latest MCC results file,
rolls out an episode with the CTRNN controller, and saves an animated GIF
showing both replicates side-by-side.

MountainCarContinuous obs: [position, velocity]
  position in [-1.2, 0.6],  goal at 0.45
  velocity in [-0.07, 0.07]

Action: continuous scalar in [-1, 1]  (from np.tanh of CTRNN output)
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
    files = sorted(glob.glob(f'{results_dir}/stage1_mountaincarcontinuous_*.json'))
    if not files:
        raise FileNotFoundError(f"No MCC results in {results_dir}/")
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
# Roll out one episode, collecting (position, velocity, action) each step
# ---------------------------------------------------------------------------

def rollout(ctrl, max_steps=999, seed=0, goal_position=0.45):
    env = gym.make('MountainCarContinuous-v0')
    env.unwrapped.goal_position = goal_position
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    frames = [(obs[0], obs[1], 0.0)]   # (pos, vel, action)
    total_reward = 0.0
    reached_at = None

    for step in range(max_steps):
        flat = np.asarray(obs, dtype=np.float32)
        logits = ctrl.forward(flat)
        action = np.tanh(logits)          # continuous: shape (1,)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append((obs[0], obs[1], float(action[0])))
        if terminated:
            reached_at = step + 1
            break
        if truncated:
            break

    env.close()
    return frames, total_reward, reached_at


# ---------------------------------------------------------------------------
# MountainCar valley geometry helpers
# ---------------------------------------------------------------------------

# The gym uses: height(x) = sin(3x) * 0.45 + 0.55
def _height(x):
    return np.sin(3.0 * x) * 0.45 + 0.55

def _slope(x):
    """Gradient of height, for car tilt angle."""
    return np.cos(3.0 * x) * 3.0 * 0.45


_VALLEY_X = np.linspace(-1.2, 0.6, 300)
_VALLEY_Y = _height(_VALLEY_X)


def draw_mcc_frame(ax, pos, vel, action,
                   car_color='#4fc3f7', goal_color='#69f0ae',
                   goal_position=0.45):
    ax.fill_between(_VALLEY_X, _VALLEY_Y - 0.3, _VALLEY_Y,
                    color='#3d5a3e', alpha=0.8, zorder=1)
    ax.plot(_VALLEY_X, _VALLEY_Y, color='#8bc34a', lw=1.5, zorder=2)

    # Goal flag
    gx = goal_position
    gy = _height(gx)
    ax.plot([gx, gx], [gy, gy + 0.12], color=goal_color, lw=2, zorder=3)
    ax.fill([gx, gx + 0.05, gx], [gy + 0.12, gy + 0.09, gy + 0.06],
            color=goal_color, zorder=3)
    ax.text(gx + 0.07, gy + 0.09, 'goal', color=goal_color, fontsize=7, zorder=5)

    # Car body — small rectangle tilted to match hill slope
    cy = _height(pos)
    angle = np.arctan(_slope(pos))
    car_w, car_h = 0.08, 0.04
    # Corners relative to car centre, pre-rotation
    corners = np.array([[-car_w/2, -car_h/2],
                         [ car_w/2, -car_h/2],
                         [ car_w/2,  car_h/2],
                         [-car_w/2,  car_h/2]])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    rotated = (rot @ corners.T).T
    xs = rotated[:, 0] + pos
    ys = rotated[:, 1] + cy + 0.02
    ax.fill(xs, ys, color=car_color, zorder=4)

    # Action arrow (thrust direction)
    arrow_len = action * 0.12
    ax.annotate('', xy=(pos + arrow_len, cy + 0.06),
                xytext=(pos, cy + 0.06),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
                zorder=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    data, results_path = load_results(path_arg)
    config = data['config']
    results = data['results']

    MAX_STEPS = config.get('MOUNTAINCARCONTINUOUS_MAX_STEPS', 999)
    SEED = 0
    GOAL_POSITION = 0.35   # hardest test variant

    all_frames = []
    infos = []
    for i, result in enumerate(results):
        if 'best_genome' not in result:
            print(f"Rep {i+1}: no best_genome — skipping")
            continue
        ctrl = decode_controller(result, config)
        frames, total_reward, reached_at = rollout(
            ctrl, max_steps=MAX_STEPS, seed=SEED, goal_position=GOAL_POSITION)
        all_frames.append(frames)
        fitness = result.get('best_fitness', '?')
        label = (f"Rep {i+1} | seed={result['seed']} | "
                 f"fit={fitness:.4f} | steps={len(frames)-1}"
                 + (f" (GOAL reached at {reached_at})" if reached_at else " (not reached)"))
        infos.append(label)
        print(label)

    if not all_frames:
        print("No replicates to animate.")
        return

    max_len = max(len(f) for f in all_frames)
    for fl in all_frames:
        while len(fl) < max_len:
            fl.append(fl[-1])

    fig, axes = plt.subplots(1, len(all_frames), figsize=(6 * len(all_frames), 4))
    if len(all_frames) == 1:
        axes = [axes]

    fig.patch.set_facecolor('#1a1a2e')

    def _setup_ax(ax):
        ax.set_xlim(-1.25, 0.65)
        ax.set_ylim(0.1, 1.4)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    for ax in axes:
        _setup_ax(ax)

    legend_patches = [
        mpatches.Patch(color='#4fc3f7', label='Car'),
        mpatches.Patch(color='#69f0ae', label='Goal (x=0.45)'),
        mpatches.Patch(color='yellow',  label='Thrust →'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               facecolor='#1a1a2e', labelcolor='white', fontsize=9,
               framealpha=0.5)

    title_obj = fig.suptitle('', fontsize=10, color='white', y=1.01)

    def init():
        for ax in axes:
            ax.cla()
            _setup_ax(ax)
        return []

    def update(frame_idx):
        init()
        for ax, frames, info in zip(axes, all_frames, infos):
            pos, vel, action = frames[frame_idx]
            draw_mcc_frame(ax, pos, vel, action, goal_position=GOAL_POSITION)
            ax.set_title(info, color='white', fontsize=7.5, pad=4)
            ax.set_xlabel(
                f'step {frame_idx}  |  pos={pos:.3f}  vel={vel:.4f}  act={action:.3f}',
                color='#aaa', fontsize=8)
        title_obj.set_text(
            f"MountainCarContinuous CTRNN — TEST: goal_position=0.35 (hardest) — "
            f"p={config.get('NEUTRALITY_P', 0)} hidden={config['HIDDEN_SIZE']}")
        return []

    ani = FuncAnimation(fig, update, frames=max_len, init_func=init,
                        interval=40, blit=False)

    out_path = Path(results_path).parent / \
        f"mcc_animation_goal035_{Path(results_path).stem}.gif"
    print(f"Saving {max_len} frames to {out_path} ...")
    ani.save(str(out_path), writer=PillowWriter(fps=25))
    print(f"Done → {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
