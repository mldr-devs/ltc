"""One-page (A4) visual summary of what the agents did in a single history file.

Every ltc.run history gets one page, so runs of different agents over the same
config can be flipped through side by side. The page mixes per-epoch aggregates
(how the network behaves over training) with a step-level action raster from one
epoch (what the individual stations actually do).
"""

from argparse import ArgumentParser
from pathlib import Path

import cloudpickle
import lz4.frame
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

from ltc.sim.constants import Actions, INITIAL_CAPACITY
from ltc.utils.history import unpack_history

A4 = (8.27, 11.69)

# Channel state as recorded by ltc.sim.sim.channel_state_selector.
IDLE, SUCCESS, COLLISION = 0, 1, -1

DRL_COLOR = 'tab:red'
LEGACY_COLOR = 'tab:blue'
# Action raster categories, in the order they are encoded below.
RASTER_LABELS = ['Idle', 'CS', 'TX ok', 'TX collided']
RASTER_CMAP = ListedColormap(['#eceff4', '#8fb8de', '#2e7d32', '#c62828'])

PAGE_PARAMS = {
    'figure.dpi': 100,
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'axes.labelsize': 7,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.35,
    'grid.linewidth': 0.4,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'lines.linewidth': 0.9,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


def load(path):
    with lz4.frame.open(path, 'rb') as f:
        _, history, metadata = unpack_history(cloudpickle.load(f))
    return history, metadata or {}


def smoothed(y, window):
    """Rolling mean along the epoch axis, keeping the series length."""
    if window <= 1 or y.shape[0] < window:
        return y
    kernel = np.ones(window) / window
    pad = ((window - 1) // 2, window // 2)
    padded = np.pad(y, (pad, *[(0, 0)] * (y.ndim - 1)), mode='edge')
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='valid'), 0, padded)


def group_means(per_station, n_drl, n):
    """Split a (epoch, station) series into its DRL and legacy group means."""
    drl = per_station[:, :n_drl].mean(axis=1) if n_drl > 0 else None
    legacy = per_station[:, n_drl:].mean(axis=1) if n_drl < n else None
    return drl, legacy


def plot_groups(ax, xs, per_station, n_drl, n, marker):
    """Both group means, with the per-station spread of each behind them."""
    for lo, hi, color in ((0, n_drl, DRL_COLOR), (n_drl, n, LEGACY_COLOR)):
        if lo >= hi:
            continue
        band = per_station[:, lo:hi]
        ax.fill_between(xs, band.min(axis=1), band.max(axis=1), color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, band.mean(axis=1), color=color, marker=marker, markersize=2.5)


def zero_based_ylim(ax, data):
    """Anchor the axis at zero with headroom, so a flat series stays visible."""
    top = float(np.max(data))
    ax.set_ylim(0, top * 1.15 if top > 0 else 1.0)


def group_legend(ax, n_drl, n, agent_label, legacy_label, extra=()):
    handles = list(extra)
    if n_drl > 0:
        handles.append(Line2D([], [], color=DRL_COLOR, label=agent_label))
    if n_drl < n:
        handles.append(Line2D([], [], color=LEGACY_COLOR, label=legacy_label))
    if handles:
        ax.legend(handles=handles, loc='best')


def stacked_shares(ax, xs, shares, labels, colors, ylabel, title):
    ax.stackplot(xs, *shares, labels=labels, colors=colors, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower center', ncol=len(labels), frameon=True, framealpha=0.85, edgecolor='none')


def draw_header(ax, path, meta, n, n_drl, n_epochs, n_steps, epoch):
    ax.axis('off')
    agent = meta.get('agent_type', '?')
    fields = [
        ('agent', f"{agent} x{n_drl}" + (f" + {meta.get('legacy_type', '?')} x{n - n_drl}" if n_drl < n else '')),
        ('traffic', str(meta.get('traffic_type', '?'))),
        ('stations', f"{n} (final {meta.get('n_final') or n})"),
        ('window', str(meta.get('window_size', '?'))),
        ('seed', str(meta.get('seed', '?'))),
        ('rollout', f"{n_epochs} x {n_steps} steps"),
        ('detail epoch', str(epoch + 1)),
    ]
    if agent == 'sr-jax' and meta.get('sr_pkl'):
        fields.append(('sr', f"{Path(meta['sr_pkl']).name} eq {meta.get('sr_eq')}"))
    if agent == 'forester' and meta.get('forest_pkl'):
        fields.append(('forest', Path(meta['forest_pkl']).name))

    ax.text(0, 1.0, Path(path).name, transform=ax.transAxes, va='top', fontsize=11, fontweight='bold')
    ax.text(
        0, 0.42,
        '   '.join(f"{k}: {v}" for k, v in fields),
        transform=ax.transAxes, va='top', fontsize=7, color='0.25',
    )


def build_page(path, output, epoch=-1, zoom_steps=200, zoom_start=0, smooth=1, n_drl=None, dpi=200):
    history, meta = load(path)

    actions = np.asarray(history.actions)
    channel = np.asarray(history.channel_state)
    rewards = np.asarray(history.rewards)
    buffers = np.asarray(history.buffer_states)
    new_frames = np.asarray(history.new_frames)
    powers = np.asarray(history.power_states)
    # `terminals` marks a station as absent, e.g. before it joins a growing network.
    live = ~np.asarray(history.terminals).astype(bool)

    n_epochs, n_steps, n = actions.shape
    # ltc.run leaves --n_drl unset when every station learns.
    n_drl = n_drl if n_drl is not None else meta.get('n_drl')
    n_drl = n if n_drl is None else min(n_drl, n)
    epoch = range(n_epochs)[epoch]

    xs = np.arange(1, n_epochs + 1)
    # A one- or two-epoch replay would otherwise draw as invisible line segments.
    marker = 'o' if n_epochs < 5 else None
    # Steps a station was actually present for, the denominator of every rate below.
    live_steps = np.maximum(live.sum(axis=1), 1)

    tx = (actions == Actions.TX.value) & live
    success = tx & (channel[..., None] == SUCCESS)
    throughput = smoothed(success.sum(axis=1) / live_steps, smooth)

    plt.rcParams.update(PAGE_PARAMS)
    fig = plt.figure(figsize=A4)
    grid = GridSpec(
        6, 2, figure=fig,
        height_ratios=[0.22, 1, 1, 1, 1, 1.15],
        hspace=0.55, wspace=0.25,
        left=0.08, right=0.97, top=0.96, bottom=0.05,
    )

    draw_header(fig.add_subplot(grid[0, :]), path, meta, n, n_drl, n_epochs, n_steps, epoch)

    # Panels sharing the epoch axis, tidied up together once they are all drawn.
    epoch_axes = []

    def epoch_ax(slot):
        ax = fig.add_subplot(slot)
        epoch_axes.append(ax)
        return ax

    # Throughput: the headline metric, per station group.
    ax = epoch_ax(grid[1, 0])
    plot_groups(ax, xs, throughput, n_drl, n, marker)
    network = smoothed(success.sum(axis=(1, 2)) / n_steps, smooth)
    ax.plot(xs, network, color='k', linestyle=':')
    ax.set_ylabel('Successful TX per step')
    ax.set_title('Throughput')
    zero_based_ylim(ax, network)
    ax.grid(True)
    group_legend(
        ax, n_drl, n, meta.get('agent_type', 'DRL'), meta.get('legacy_type', 'legacy'),
        extra=[Line2D([], [], color='k', linestyle=':', label='Network')],
    )

    # Where the channel time goes -- the collision rate is the cost of the policy.
    ax = epoch_ax(grid[1, 1])
    occupancy = [(channel == state).mean(axis=1) for state in (IDLE, SUCCESS, COLLISION)]
    stacked_shares(
        ax, xs, occupancy, ['Idle', 'Success', 'Collision'],
        ['#eceff4', '#2e7d32', '#c62828'], 'Share of steps', 'Channel occupancy',
    )

    # The policy itself: how the agents split their steps between the three actions.
    ax = epoch_ax(grid[2, 0])
    mix = [
        ((actions[:, :, :n_drl] == a.value) & live[:, :, :n_drl]).sum(axis=(1, 2)) / live_steps[:, :n_drl].sum(axis=1)
        for a in (Actions.TX, Actions.CS, Actions.IDLE)
    ]
    stacked_shares(
        ax, xs, mix, ['TX', 'CS', 'Idle'],
        ['#c62828', '#8fb8de', '#eceff4'], 'Share of steps', f"Action mix ({meta.get('agent_type', 'DRL')})",
    )

    # Reward, the signal the agents were actually optimising.
    ax = epoch_ax(grid[2, 1])
    plot_groups(ax, xs, smoothed(rewards.mean(axis=1), smooth), n_drl, n, marker)
    ax.set_ylabel('Mean reward per step')
    ax.set_title('Reward')
    ax.grid(True)

    # Backlog and the delay it implies; flat lines here mean the buffers keep up.
    ax = epoch_ax(grid[3, 0])
    occupancy_series = smoothed((buffers * live).sum(axis=1) / live_steps, smooth)
    plot_groups(ax, xs, occupancy_series, n_drl, n, marker)
    ax.set_ylabel('Mean buffer occupancy')
    ax.set_title('Buffer')
    zero_based_ylim(ax, occupancy_series)
    ax.grid(True)

    ax = epoch_ax(grid[3, 1])
    delay = (buffers * live).sum(axis=1) / np.maximum((new_frames * live).sum(axis=1), 1e-6)
    delay = smoothed(delay, smooth)
    plot_groups(ax, xs, delay, n_drl, n, marker)
    ax.set_ylabel('Steps per frame')
    ax.set_title('Channel access delay')
    zero_based_ylim(ax, delay)
    ax.grid(True)

    # Fairness across all stations: 1.0 means the successful transmissions were
    # shared evenly, 1/n means a single station monopolised the channel.
    ax = epoch_ax(grid[4, 0])
    per_station = success.sum(axis=1).astype(float)
    fairness = per_station.sum(axis=1) ** 2 / np.maximum(n * (per_station ** 2).sum(axis=1), 1e-9)
    ax.plot(xs, smoothed(fairness, smooth), color='k', marker=marker, markersize=2.5)
    ax.axhline(1.0, color='0.6', linestyle=':')
    ax.set_ylabel("Jain's index")
    ax.set_xlabel('Epoch')
    ax.set_title('Fairness')
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # Battery drain, the other half of the reward's trade-off.
    ax = epoch_ax(grid[4, 1])
    consumed = (INITIAL_CAPACITY - powers[:, -1]) / INITIAL_CAPACITY
    plot_groups(ax, xs, consumed, n_drl, n, marker)
    ax.set_ylabel('Consumed power')
    ax.set_xlabel('Epoch')
    ax.set_title('Power')
    zero_based_ylim(ax, consumed)
    ax.grid(True)

    # Step-level view of one epoch: what each station was doing, slot by slot.
    ax = fig.add_subplot(grid[5, :])
    stop = min(zoom_start + zoom_steps, n_steps)
    window = slice(zoom_start, stop)
    raster = np.where(actions[epoch, window] == Actions.CS.value, 1, 0)
    raster = np.where(actions[epoch, window] == Actions.TX.value, 2, raster)
    raster = np.where(
        (actions[epoch, window] == Actions.TX.value) & (channel[epoch, window, None] != SUCCESS), 3, raster,
    )
    ax.imshow(
        raster.T, aspect='auto', interpolation='nearest', cmap=RASTER_CMAP, vmin=0, vmax=3,
        extent=(zoom_start, stop, n - 0.5, -0.5),
    )
    if 0 < n_drl < n:
        ax.axhline(n_drl - 0.5, color='k', linewidth=0.8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f"{i}{'*' if i < n_drl else ''}" for i in range(n)])
    ax.set_xlabel(f'Step within epoch {epoch + 1}')
    ax.set_ylabel('Station')
    ax.set_title(f'Per-station activity, steps {zoom_start}-{stop} (* = {meta.get("agent_type", "DRL")})')
    ax.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(RASTER_CMAP.colors, RASTER_LABELS)],
        loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=4,
    )

    for ax in epoch_axes:
        # Epochs are whole numbers; a short rollout would otherwise get 1.25, 1.50, ...
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if n_epochs > 1:
            ax.set_xlim(1, n_epochs)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = ArgumentParser(description='Render a one-page A4 summary of a history file.')
    parser.add_argument('--file', type=str, required=True, help='Path to the history .pkl.lz4 file.')
    parser.add_argument('--output', type=str, help='Output page path. Defaults to <history>.page.pdf.')
    parser.add_argument('--epoch', type=int, default=-1, help='Epoch shown in the per-station activity raster.')
    parser.add_argument('--zoom_steps', type=int, default=200, help='Steps covered by the activity raster.')
    parser.add_argument('--zoom_start', type=int, default=0, help='First step of the activity raster.')
    parser.add_argument('--smooth', type=int, default=1, help='Rolling-mean window, in epochs, for the curves.')
    parser.add_argument('--n_drl', type=int, help='Number of learning stations. Defaults to the value in the history.')
    parser.add_argument('--dpi', type=int, default=200, help='Raster resolution of the saved page.')
    args = parser.parse_args()

    mpl.use('Agg')
    output = args.output or f"{args.file.removesuffix('.pkl.lz4')}.page.pdf"
    print(f"Saved: {build_page(args.file, output, args.epoch, args.zoom_steps, args.zoom_start, args.smooth, args.n_drl, args.dpi)}")
