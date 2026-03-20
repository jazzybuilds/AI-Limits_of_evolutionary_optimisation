"""
CTRNN Network Structure Visualizer
===================================
Loads the best genome from each replicate in a Stage 1 results file,
decodes it, and plots:
  1. Weight matrix heatmaps (W_in, W_recurrent, W_out)
  2. Time constants (tau) and biases per neuron
  3. Network graph with edge weights as thickness/colour
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file, no window needed
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

from matplotlib.gridspec import GridSpec
from genome_decoder import build_decoder_for_controller


def load_latest_results(results_dir='results_stage1'):
    files = sorted(glob.glob(f'{results_dir}/*.json'))
    if not files:
        raise FileNotFoundError(f"No results found in {results_dir}/")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f), path


def decode_best_ctrnn(result, config):
    """Decode best genome from a replicate result dict."""
    genome = np.array(result['best_genome'], dtype=int)
    decoder = build_decoder_for_controller(
        controller_type='ctrnn',
        obs_dim=result['obs_dim'],
        act_dim=result['act_dim'],
        hidden_size=config['HIDDEN_SIZE'],
        bits_per_weight=config['BITS_PER_WEIGHT']
    )
    params = decoder.decode(genome)
    H = config['HIDDEN_SIZE']
    obs_dim = result['obs_dim']
    act_dim = result['act_dim']
    return {
        'W':     params['W'].reshape(H, H),
        'W_in':  params['W_in'].reshape(H, obs_dim),
        'W_out': params['W_out'].reshape(act_dim, H),
        'bias':  params['bias'],
        'tau':   params['tau'],
    }, H, obs_dim, act_dim


def plot_weight_matrices(ax_win, ax_w, ax_wout, ax_tau, ax_bias,
                         p, H, obs_dim, act_dim, title_prefix):
    """Fill axes with heatmaps for one replicate."""
    cmap_div = 'RdBu_r'
    vmax = 3.0

    # W_in
    im = ax_win.imshow(p['W_in'], cmap=cmap_div, vmin=-vmax, vmax=vmax,
                       aspect='auto')
    ax_win.set_title(f'{title_prefix}\nInput weights W_in ({H}×{obs_dim})',
                     fontsize=9)
    ax_win.set_xlabel('Input neuron', fontsize=8)
    ax_win.set_ylabel('Hidden neuron', fontsize=8)
    plt.colorbar(im, ax=ax_win, fraction=0.046)

    # W recurrent
    im2 = ax_w.imshow(p['W'], cmap=cmap_div, vmin=-vmax, vmax=vmax)
    ax_w.set_title(f'Recurrent weights W ({H}×{H})', fontsize=9)
    ax_w.set_xlabel('From neuron', fontsize=8)
    ax_w.set_ylabel('To neuron', fontsize=8)
    plt.colorbar(im2, ax=ax_w, fraction=0.046)

    # W_out
    im3 = ax_wout.imshow(p['W_out'], cmap=cmap_div, vmin=-vmax, vmax=vmax,
                          aspect='auto')
    ax_wout.set_title(f'Output weights W_out ({act_dim}×{H})', fontsize=9)
    ax_wout.set_xlabel('Hidden neuron', fontsize=8)
    ax_wout.set_ylabel('Action', fontsize=8)
    plt.colorbar(im3, ax=ax_wout, fraction=0.046)

    # Tau (time constants)
    ax_tau.bar(range(H), p['tau'], color='steelblue', edgecolor='white')
    ax_tau.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
                   label='τ=1')
    ax_tau.set_title('Time constants τ', fontsize=9)
    ax_tau.set_xlabel('Hidden neuron', fontsize=8)
    ax_tau.set_ylabel('τ value', fontsize=8)
    ax_tau.set_ylim(0, 2.1)
    ax_tau.legend(fontsize=7)

    # Bias
    colours = ['tomato' if b < 0 else 'mediumseagreen' for b in p['bias']]
    ax_bias.bar(range(H), p['bias'], color=colours, edgecolor='white')
    ax_bias.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_bias.set_title('Neuron biases', fontsize=9)
    ax_bias.set_xlabel('Hidden neuron', fontsize=8)
    ax_bias.set_ylabel('Bias value', fontsize=8)


def plot_network_graph(ax, p, H, obs_dim, act_dim, title, env_type='cartpole'):
    """
    Draw a schematic network graph.
    For large obs_dim (MiniGrid), shows hidden+output only with compact layout.
    For CartPole (obs_dim<=20), shows full input->hidden->output graph.
    Edge colour/alpha encodes weight sign/magnitude.
    """
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')

    node_r = 0.06
    cmap = cm.RdBu_r
    norm = mcolors.Normalize(vmin=-3, vmax=3)

    if obs_dim > 20:
        # Compact mode: skip input nodes (too many to draw legibly)
        ax.set_xlim(-0.2, 2.5)
        ax.set_ylim(-0.1, 1.1)

        hid_pos = [(1.0, y) for y in np.linspace(0.1, 0.9, H)]
        out_pos = [(2.0, y) for y in np.linspace(0.3, 0.7, act_dim)]

        # Recurrent edges (hidden <-> hidden)
        for j, (x2, y2) in enumerate(hid_pos):
            for i, (x1, y1) in enumerate(hid_pos):
                if i == j:
                    continue
                w = p['W'][j, i]
                if abs(w) < 0.5:
                    continue
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color=cmap(norm(w)),
                                            lw=min(abs(w) * 0.8, 2.0), alpha=0.5),
                            zorder=1)

        # Hidden -> output edges
        for j, (x2, y2) in enumerate(out_pos):
            for i, (x1, y1) in enumerate(hid_pos):
                w = p['W_out'][j, i]
                if abs(w) < 0.3:
                    continue
                ax.plot([x1, x2], [y1, y2],
                        color=cmap(norm(w)),
                        linewidth=min(abs(w) * 1.2, 3.0),
                        alpha=0.7, zorder=1)

        # Hidden nodes (coloured by tau: cool=fast, warm=slow)
        for i, ((x, y), tau) in enumerate(zip(hid_pos, p['tau'])):
            nc = cm.coolwarm(1 - (tau - 0.1) / 1.9)
            ax.add_patch(plt.Circle((x, y), node_r, color=nc, zorder=3))
            ax.text(x, y, f'h{i}', ha='center', va='center',
                    fontsize=6, color='white', zorder=4)
            ax.text(x, y - node_r - 0.05, f'τ={tau:.2f}',
                    ha='center', fontsize=5, color='dimgray', zorder=4)

        # Output nodes
        act_labels = ['L', 'R', 'Fwd', 'Pu', 'Dr', 'To', 'Do'][:act_dim]
        for (x, y), lbl in zip(out_pos, act_labels):
            ax.add_patch(plt.Circle((x, y), node_r, color='#e15759', zorder=3))
            ax.text(x, y, lbl, ha='center', va='center',
                    fontsize=6, color='white', zorder=4)

        ax.text(0.5, -0.06,
                f'Input: {obs_dim} features (see spatial sensitivity maps left)',
                ha='center', fontsize=8, color='gray', transform=ax.transAxes)
        ax.text(0.40, 1.02, 'Hidden', ha='center', fontsize=8, transform=ax.transAxes)
        ax.text(0.82, 1.02, 'Output', ha='center', fontsize=8, transform=ax.transAxes)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label='Weight value')
        return

    # --- Full graph for small obs_dim (CartPole) ---
    ax.set_xlim(-0.5, 3.5)
    ax.set_aspect('equal')

    # Node positions
    def col_positions(n, x):
        ys = np.linspace(0.1, 0.9, n)
        return [(x, y) for y in ys]

    in_pos  = col_positions(obs_dim, 0.5)
    hid_pos = col_positions(H, 2.0)
    out_pos = col_positions(act_dim, 3.0)

    def draw_edges(src_pos, dst_pos, W, threshold=0.3):
        for j, (x2, y2) in enumerate(dst_pos):
            for i, (x1, y1) in enumerate(src_pos):
                w = W[j, i]
                if abs(w) < threshold:
                    continue
                colour = cmap(norm(w))
                lw = min(abs(w) * 1.2, 3.0)
                ax.plot([x1, x2], [y1, y2], color=colour,
                        linewidth=lw, alpha=0.7, zorder=1)

    # Draw recurrent edges (self/lateral within hidden layer)
    for j, (x2, y2) in enumerate(hid_pos):
        for i, (x1, y1) in enumerate(hid_pos):
            if i == j:
                continue
            w = p['W'][j, i]
            if abs(w) < 0.5:
                continue
            colour = cmap(norm(w))
            lw = min(abs(w) * 0.8, 2.0)
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=colour,
                                        lw=lw, alpha=0.5),
                        zorder=1)

    draw_edges(in_pos,  hid_pos, p['W_in'],  threshold=0.3)
    draw_edges(hid_pos, out_pos, p['W_out'], threshold=0.3)

    # Draw nodes
    in_labels  = [f'x{i}' for i in range(obs_dim)]
    hid_labels = [f'h{i}' for i in range(H)]
    out_labels = ['Left', 'Right'] if act_dim == 2 else [f'a{i}' for i in range(act_dim)]

    for (x, y), lbl in zip(in_pos, in_labels):
        c = plt.Circle((x, y), node_r, color='#4e79a7', zorder=3)
        ax.add_patch(c)
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=6, color='white', zorder=4)

    for (x, y), lbl, tau in zip(hid_pos, hid_labels, p['tau']):
        # Colour hidden nodes by tau: warm=slow, cool=fast
        t_norm = (tau - 0.1) / 1.9
        nc = cm.coolwarm(1 - t_norm)
        c = plt.Circle((x, y), node_r, color=nc, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=6, color='white', zorder=4)
        ax.text(x, y - node_r - 0.05, f'τ={tau:.2f}',
                ha='center', fontsize=5, color='dimgray', zorder=4)

    for (x, y), lbl in zip(out_pos, out_labels):
        c = plt.Circle((x, y), node_r, color='#e15759', zorder=3)
        ax.add_patch(c)
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=6, color='white', zorder=4)

    # Column labels
    ax.text(0.5,  0.97, 'Inputs',  ha='center', fontsize=8,
            transform=ax.transAxes)
    ax.text(0.45, 0.97, 'Hidden',  ha='center', fontsize=8,
            transform=ax.transAxes)
    ax.text(0.95, 0.97, 'Outputs', ha='center', fontsize=8,
            transform=ax.transAxes)

    # Colourbar legend
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04,
                 label='Weight value')


def plot_win_spatial(ax1, ax2, ax3, W_in, H):
    """
    For MiniGrid: visualize W_in as spatial sensitivity maps.
    Assumes 7x7x3 partial observation (obs_dim=147).
    Shows mean |W_in| across hidden neurons per feature channel.
    Channels: object type, colour index, object state.
    """
    if W_in.shape[1] != 147:
        for ax in (ax1, ax2, ax3):
            ax.text(0.5, 0.5, f'obs_dim={W_in.shape[1]}\n(not 7×7×3)',
                    ha='center', va='center', transform=ax.transAxes)
        return

    # Mean absolute sensitivity per spatial position per channel: (7, 7, 3)
    sens = np.abs(W_in).reshape(H, 7, 7, 3).mean(axis=0)
    channel_names = ['Object Type', 'Color', 'State']
    for ax, ch, name in zip((ax1, ax2, ax3), range(3), channel_names):
        im = ax.imshow(sens[:, :, ch], cmap='hot', vmin=0)
        ax.set_title(f'W_in sensitivity\n({name})', fontsize=8)
        ax.set_xlabel('Grid col', fontsize=7)
        ax.set_ylabel('Grid row', fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    import sys
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        print(f"Loading: {results_path}")
        with open(results_path) as f:
            data = json.load(f)
    else:
        data, results_path = load_latest_results()
    config = data['config']
    results = data['results']
    env_type = config.get('ENV_TYPE', 'cartpole')

    n_rep = len(results)
    fig = plt.figure(figsize=(20, 8 * n_rep))
    fig.suptitle(
        f"Best CTRNN Neural Structure — {config['ENV_TYPE'].upper()} "
        f"[p={config.get('NEUTRALITY_P', 0.0)}, "
        f"hidden={config['HIDDEN_SIZE']}]",
        fontsize=13, fontweight='bold', y=1.01
    )

    gs = GridSpec(n_rep * 2, 5, figure=fig, hspace=0.5, wspace=0.4)

    for rep_idx, result in enumerate(results):
        if 'best_genome' not in result:
            print(f"Replicate {rep_idx+1}: no best_genome saved — skipping")
            continue

        p, H, obs_dim, act_dim = decode_best_ctrnn(result, config)
        fitness = result.get('best_fitness', result['final_stats']['max_fitness'])
        prefix = (f"Replicate {rep_idx+1} | seed={result['seed']} | "
                  f"best fitness={fitness:.4f}")

        row1 = rep_idx * 2
        row2 = rep_idx * 2 + 1

        ax_win  = fig.add_subplot(gs[row1, 0])
        ax_w    = fig.add_subplot(gs[row1, 1])
        ax_wout = fig.add_subplot(gs[row1, 2])
        ax_tau  = fig.add_subplot(gs[row1, 3])
        ax_bias = fig.add_subplot(gs[row1, 4])

        plot_weight_matrices(ax_win, ax_w, ax_wout, ax_tau, ax_bias,
                              p, H, obs_dim, act_dim, prefix)

        if env_type == 'minigrid' and obs_dim > 20:
            # Row 2: spatial W_in maps (cols 0-2) + compact network graph (cols 3-4)
            ax_ch1 = fig.add_subplot(gs[row2, 0])
            ax_ch2 = fig.add_subplot(gs[row2, 1])
            ax_ch3 = fig.add_subplot(gs[row2, 2])
            ax_net = fig.add_subplot(gs[row2, 3:])
            plot_win_spatial(ax_ch1, ax_ch2, ax_ch3, p['W_in'], H)
        else:
            # Row 2: full-width network graph (CartPole)
            ax_net = fig.add_subplot(gs[row2, :])

        plot_network_graph(ax_net, p, H, obs_dim, act_dim,
                           f'Network graph — {prefix}',
                           env_type=env_type)

    plt.tight_layout()
    out_path = Path(results_path).parent / \
        f"ctrnn_structure_{Path(results_path).stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
