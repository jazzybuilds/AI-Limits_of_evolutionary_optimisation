"""
LunarLander CTRNN Animation
============================
Loads the best genome from the best replicate of the latest LunarLander
results file, rolls out several episodes, picks the best one, and saves
an animated GIF using gymnasium's rgb_array render mode.

Usage:
    python animate_lunarlander.py                          # auto-find latest
    python animate_lunarlander.py results_stage1/stage1_lunarlander_ctrnn_1773897161.json
"""
import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    files = sorted(glob.glob(f'{results_dir}/stage1_lunarlander_ctrnn_[0-9]*.json'))
    if not files:
        raise FileNotFoundError(f"No LunarLander results in {results_dir}/")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f), path


def decode_controller(result, config):
    encoding = config.get('GENOME_ENCODING', 'binary')
    decoder = build_decoder_for_controller(
        controller_type='ctrnn',
        obs_dim=result['obs_dim'],
        act_dim=result['act_dim'],
        hidden_size=config['HIDDEN_SIZE'],
        bits_per_weight=config.get('BITS_PER_WEIGHT', 16),
    )
    if encoding == 'realvalued':
        genome = np.array(result['best_genome'], dtype=np.float64)
        params = decoder.decode_realvalued(genome)
    else:
        genome = np.array(result['best_genome'], dtype=int)
        params = decoder.decode(genome)

    ctrl = build_controller(
        'ctrnn', result['obs_dim'], result['act_dim'],
        config['HIDDEN_SIZE'], dt=config.get('CTRNN_DT', 0.2)
    )
    ctrl.set_params(params)
    return ctrl


# ---------------------------------------------------------------------------
# Roll out one episode using rendered frames
# ---------------------------------------------------------------------------

def rollout(ctrl, env_kwargs=None, max_steps=500, seed=0, continuous=False):
    if env_kwargs is None:
        env_kwargs = {}
    env_id = 'LunarLanderContinuous-v3' if continuous else 'LunarLander-v3'
    env = gym.make(env_id, render_mode='rgb_array', **env_kwargs)
    obs, _ = env.reset(seed=seed)
    ctrl.reset()

    frames = [env.render()]
    total_reward = 0.0
    landed = False

    for step in range(max_steps):
        flat = np.asarray(obs, dtype=np.float32)
        logits = ctrl.forward(flat)
        if continuous:
            action = np.tanh(np.asarray(logits, dtype=np.float32))
        else:
            action = int(np.argmax(logits))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
        if terminated or truncated:
            if terminated and total_reward > 0:
                landed = True
            break

    env.close()
    return frames, total_reward, landed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data, fpath = load_results(path)
    config = data['config']

    # Pick the replicate with the highest best_fitness
    best_rep = max(data['results'], key=lambda r: r['best_fitness'])
    print(f"Best replicate: {best_rep['replicate']} | "
          f"fitness={best_rep['best_fitness']:.4f} | seed={best_rep['seed']}")

    ctrl = decode_controller(best_rep, config)

    env_type = config.get('ENV_TYPE', 'lunarlander')
    continuous = env_type == 'lunarlander_continuous'

    # Try several seeds, keep the best episode (highest reward)
    best_frames = None
    best_reward = -np.inf
    best_landed = False

    print("Rolling out episodes to find best...")
    for seed in range(10):
        frames, reward, landed = rollout(ctrl, max_steps=500, seed=seed, continuous=continuous)
        print(f"  seed={seed:2d}  reward={reward:8.1f}  steps={len(frames)-1}"
              f"  {'LANDED' if landed else ''}")
        if reward > best_reward:
            best_reward = reward
            best_frames = frames
            best_landed = landed

    print(f"\nBest episode: reward={best_reward:.1f} "
          f"{'— LANDED!' if best_landed else '— did not land'}")

    # --- How close to landing? ---
    # fitness = clip((reward + 100) / 300, 0, 1)
    # landing threshold = reward > 0 → fitness > 0.333
    fitness = float(np.clip((best_reward + 100) / 300, 0, 1))
    landing_threshold = 200   # gym's nominal "solved" reward
    gap = landing_threshold - best_reward
    print(f"Fitness: {fitness:.3f} | Gap to landing (+0 reward): {-best_reward:.0f} points")
    print(f"Gap to 'solved' (+200 reward): {gap:.0f} points")

    # --- Save GIF ---
    out_stem = Path(fpath).stem
    out_path = Path('results_stage1') / f'lunarlander_best_{out_stem}.gif'

    fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
    ax.axis('off')
    im = ax.imshow(best_frames[0])
    title = ax.set_title('', fontsize=10)

    def update(i):
        im.set_data(best_frames[i])
        status = 'LANDED!' if (best_landed and i == len(best_frames) - 1) else ''
        title.set_text(f'Step {i} | Reward so far ≈ {best_reward:.0f}  {status}')
        return [im, title]

    # Sample every 2nd frame to keep file size manageable
    frame_indices = list(range(0, len(best_frames), 2))
    anim = FuncAnimation(
        fig, update, frames=frame_indices,
        interval=40, blit=True
    )

    print(f"\nSaving GIF ({len(frame_indices)} frames) → {out_path}")
    anim.save(str(out_path), writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f"Done: {out_path}")


if __name__ == '__main__':
    main()
