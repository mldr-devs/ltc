"""One-page (A4) visual summary of what the agents did in a single rollout.

Every ltc.run history gets one page, so runs of different agents over the same
config can be flipped through side by side.

The whole page describes *one* rollout, binned into step windows: the curves and
the action raster below them are two views of the same stretch of simulated time,
and the raster's span is shaded on every curve. A replay is one epoch, so that is
the whole run; a training run has many, and the page draws the last one, which is
the converged policy. Learning progress is deliberately not on this page -- the
epoch axis used to be the x axis here, which meant the curves described the whole
of training while the raster underneath came from a single epoch, and the two
could not be read against each other.
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
from matplotlib.patches import Patch

from ltc.sim.constants import Actions, INITIAL_CAPACITY
from ltc.utils.history import unpack_history
from ltc.utils.metrics import success_mask

A4 = (8.27, 11.69)

# Channel state as recorded by ltc.sim.sim.channel_state_selector.
IDLE, SUCCESS, COLLISION = 0, 1, -1

DRL_COLOR = 'tab:red'
LEGACY_COLOR = 'tab:blue'
ZOOM_COLOR = '#f0c000'
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


def blocks(y, window):
    """Reshape the leading step axis into ``[n_windows, window, ...]``.

    Every curve on the page is a reduction over this: a sum for counts, a mean for
    levels. A trailing partial window is dropped rather than averaged over fewer
    steps, which would make the last point noisier than the rest for no reason.
    """
    n_windows = y.shape[0] // window
    return y[:n_windows * window].reshape(n_windows, window, *y.shape[1:])


def smoothed(y, window):
    """Rolling mean along the window axis, keeping the series length."""
    if window <= 1 or y.shape[0] < window:
        return y
    kernel = np.ones(window) / window
    pad = ((window - 1) // 2, window // 2)
    padded = np.pad(y, (pad, *[(0, 0)] * (y.ndim - 1)), mode='edge')
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='valid'), 0, padded)


def plot_groups(ax, xs, per_station, n_drl, n, marker):
    """Both group means, with the per-station spread of each behind them."""
    for lo, hi, color in ((0, n_drl, DRL_COLOR), (n_drl, n, LEGACY_COLOR)):
        if lo >= hi:
            continue
        band = per_station[:, lo:hi]
        # nan-aware: the delay series leaves gaps where no frame arrived.
        ax.fill_between(
            xs, np.nanmin(band, axis=1), np.nanmax(band, axis=1), color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(xs, np.nanmean(band, axis=1), color=color, marker=marker, markersize=2.5)


def zero_based_ylim(ax, data):
    """Anchor the axis at zero with headroom, so a flat series stays visible."""
    top = float(np.nanmax(data)) if np.size(data) and not np.all(np.isnan(data)) else 0.0
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


def draw_header(ax, path, meta, n, n_drl, n_epochs, n_steps, epoch, window):
    ax.axis('off')
    agent = meta.get('agent_type', '?')
    drawn = f"epoch {epoch + 1} of {n_epochs}" if n_epochs > 1 else 'the whole run'
    fields = [
        ('agent', f"{agent} x{n_drl}" + (f" + {meta.get('legacy_type', '?')} x{n - n_drl}" if n_drl < n else '')),
        ('traffic', str(meta.get('traffic_type', '?'))),
        ('stations', f"{n} (final {meta.get('n_final') or n})"),
        ('window', str(meta.get('window_size', '?'))),
        ('seed', str(meta.get('seed', '?'))),
        ('rollout', f"{n_steps} steps, drawn from {drawn}"),
        ('binning', f"{window} steps per point"),
    ]
    if agent == 'sr-jax' and meta.get('sr_pkl'):
        fields.append(('sr', f"{Path(meta['sr_pkl']).name} eq {meta.get('sr_eq') or 'best'}"))
    if agent == 'forester' and meta.get('forest_pkl'):
        fields.append(('forest', Path(meta['forest_pkl']).name))

    ax.text(0, 1.0, Path(path).name, transform=ax.transAxes, va='top', fontsize=11, fontweight='bold')
    # Two lines: the single line ran off the right edge of the A4 page once the
    # binning field was added.
    half = (len(fields) + 1) // 2
    for row, chunk in enumerate((fields[:half], fields[half:])):
        ax.text(
            0, 0.50 - 0.38 * row,
            '   '.join(f"{k}: {v}" for k, v in chunk),
            transform=ax.transAxes, va='top', fontsize=7, color='0.25',
        )


def build_page(path, output, epoch=-1, zoom_steps=200, zoom_start=0, smooth=1, n_drl=None,
               window=0, dpi=200):
    history, meta = load(path)

    n_epochs, n_steps, n = np.asarray(history.actions).shape
    # ltc.run leaves --n_drl unset when every station learns.
    n_drl = n_drl if n_drl is not None else meta.get('n_drl')
    n_drl = n if n_drl is None else min(n_drl, n)
    epoch = range(n_epochs)[epoch]
    # ~100 points across the rollout reads well at A4 width.
    window = window if window > 0 else max(1, n_steps // 100)

    # One rollout, and every panel below is a reduction of these. A replay has a
    # single epoch, so this picks the whole of it; a training run has many and this
    # picks the converged one.
    actions = np.asarray(history.actions)[epoch]                    # [n_steps, n]
    channel = np.asarray(history.channel_state)[epoch]              # [n_steps]
    rewards = np.asarray(history.rewards)[epoch]
    buffers = np.asarray(history.buffer_states)[epoch]
    new_frames = np.asarray(history.new_frames)[epoch]
    powers = np.asarray(history.power_states)[epoch]
    # `terminals` marks a station as absent, e.g. before it joins a growing network.
    live = ~np.asarray(history.terminals)[epoch].astype(bool)

    n_windows = n_steps // window
    # Window centres, so a point sits in the middle of the steps it summarises.
    xs = np.arange(n_windows) * window + window / 2
    # A very coarse binning would otherwise draw as a couple of invisible segments.
    marker = 'o' if n_windows < 5 else None
    # Steps a station was actually present for, the denominator of every rate below.
    live_steps = np.maximum(blocks(live, window).sum(axis=1), 1)     # [n_windows, n]

    success = success_mask(actions, buffers, channel, live=live, tx_action=Actions.TX.value)
    throughput = smoothed(blocks(success, window).sum(axis=1) / live_steps, smooth)

    plt.rcParams.update(PAGE_PARAMS)
    fig = plt.figure(figsize=A4)
    grid = GridSpec(
        6, 2, figure=fig,
        height_ratios=[0.22, 1, 1, 1, 1, 1.15],
        hspace=0.55, wspace=0.25,
        left=0.08, right=0.97, top=0.96, bottom=0.05,
    )

    draw_header(fig.add_subplot(grid[0, :]), path, meta, n, n_drl, n_epochs, n_steps, epoch, window)

    # Panels sharing the step axis, tidied up together once they are all drawn.
    step_axes = []

    def step_ax(slot):
        ax = fig.add_subplot(slot)
        step_axes.append(ax)
        return ax

    # Throughput: the headline metric, per station group.
    ax = step_ax(grid[1, 0])
    plot_groups(ax, xs, throughput, n_drl, n, marker)
    network = smoothed(blocks(success.sum(axis=1), window).mean(axis=1), smooth)
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
    # SUCCESS here is the raw channel state, so unlike the panel above it counts a
    # transmission on an empty buffer: the slot was busy and uncontended either way.
    ax = step_ax(grid[1, 1])
    occupancy = [blocks(channel == state, window).mean(axis=1) for state in (IDLE, SUCCESS, COLLISION)]
    stacked_shares(
        ax, xs, occupancy, ['Idle', 'Busy', 'Collision'],
        ['#eceff4', '#2e7d32', '#c62828'], 'Share of steps', 'Channel occupancy',
    )

    # The policy itself: how the agents split their steps between the three actions.
    ax = step_ax(grid[2, 0])
    drl_live_steps = np.maximum(blocks(live[:, :n_drl].sum(axis=1), window).sum(axis=1), 1)
    mix = [
        blocks(((actions[:, :n_drl] == a.value) & live[:, :n_drl]).sum(axis=1), window).sum(axis=1)
        / drl_live_steps
        for a in (Actions.TX, Actions.CS, Actions.IDLE)
    ]
    stacked_shares(
        ax, xs, mix, ['TX', 'CS', 'Idle'],
        ['#c62828', '#8fb8de', '#eceff4'], 'Share of steps', f"Action mix ({meta.get('agent_type', 'DRL')})",
    )

    # Reward, the signal the agents were actually optimising.
    ax = step_ax(grid[2, 1])
    plot_groups(ax, xs, smoothed(blocks(rewards, window).mean(axis=1), smooth), n_drl, n, marker)
    ax.set_ylabel('Mean reward per step')
    ax.set_title('Reward')
    ax.grid(True)

    # Backlog and the delay it implies; flat lines here mean the buffers keep up.
    ax = step_ax(grid[3, 0])
    occupancy_series = smoothed(blocks(buffers * live, window).sum(axis=1) / live_steps, smooth)
    plot_groups(ax, xs, occupancy_series, n_drl, n, marker)
    ax.set_ylabel('Mean buffer occupancy')
    ax.set_title('Buffer')
    zero_based_ylim(ax, occupancy_series)
    ax.grid(True)

    ax = step_ax(grid[3, 1])
    arrivals = blocks(new_frames * live, window).sum(axis=1)
    # Backlog per arriving frame is undefined in a window nothing arrived in, and
    # at these window sizes that happens often on bursty traffic. Dividing by a
    # floor instead turned those windows into 2e7 spikes that flattened the rest of
    # the series into the axis; NaN leaves an honest gap in the line.
    delay = np.where(
        arrivals > 0, blocks(buffers * live, window).sum(axis=1) / np.maximum(arrivals, 1e-9), np.nan,
    )
    delay = smoothed(delay, smooth)
    plot_groups(ax, xs, delay, n_drl, n, marker)
    ax.set_ylabel('Steps per frame')
    ax.set_title('Channel access delay')
    zero_based_ylim(ax, delay)
    ax.grid(True)

    # Fairness across all stations: 1.0 means the successful transmissions were
    # shared evenly within the window, 1/n means a single station monopolised it.
    ax = step_ax(grid[4, 0])
    per_station = blocks(success, window).sum(axis=1).astype(float)
    fairness = per_station.sum(axis=1) ** 2 / np.maximum(n * (per_station ** 2).sum(axis=1), 1e-9)
    ax.plot(xs, smoothed(fairness, smooth), color='k', marker=marker, markersize=2.5)
    ax.axhline(1.0, color='0.6', linestyle=':')
    ax.set_ylabel("Jain's index")
    ax.set_xlabel('Step')
    ax.set_title('Fairness')
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # Battery drain, the other half of the reward's trade-off. Monotone by
    # construction, so its slope is the reading: how fast the policy spends power.
    ax = step_ax(grid[4, 1])
    consumed = smoothed(blocks((INITIAL_CAPACITY - powers) / INITIAL_CAPACITY, window).mean(axis=1), smooth)
    plot_groups(ax, xs, consumed, n_drl, n, marker)
    ax.set_ylabel('Consumed power')
    ax.set_xlabel('Step')
    ax.set_title('Power')
    zero_based_ylim(ax, consumed)
    ax.grid(True)

    # Step-level view of the same rollout: what each station was doing, slot by
    # slot, over the stretch shaded on every curve above.
    ax = fig.add_subplot(grid[5, :])
    stop = min(zoom_start + zoom_steps, n_steps)
    zoom = slice(zoom_start, stop)
    raster = np.where(actions[zoom] == Actions.CS.value, 1, 0)
    raster = np.where(actions[zoom] == Actions.TX.value, 2, raster)
    raster = np.where(
        (actions[zoom] == Actions.TX.value) & (channel[zoom, None] != SUCCESS), 3, raster,
    )
    ax.imshow(
        raster.T, aspect='auto', interpolation='nearest', cmap=RASTER_CMAP, vmin=0, vmax=3,
        extent=(zoom_start, stop, n - 0.5, -0.5),
    )
    if 0 < n_drl < n:
        ax.axhline(n_drl - 0.5, color='k', linewidth=0.8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f"{i}{'*' if i < n_drl else ''}" for i in range(n)])
    ax.set_xlabel('Step')
    ax.set_ylabel('Station')
    ax.set_title(f'Per-station activity, steps {zoom_start}-{stop} (* = {meta.get("agent_type", "DRL")})')
    ax.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(RASTER_CMAP.colors, RASTER_LABELS)],
        loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=4,
    )

    for ax in step_axes:
        ax.set_xlim(0, n_windows * window)
        # Tie the raster to the curves: this is the stretch drawn in full below.
        ax.axvspan(zoom_start, stop, color=ZOOM_COLOR, alpha=0.18, linewidth=0, zorder=0)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = ArgumentParser(description='Render a one-page A4 summary of one rollout of a history file.')
    parser.add_argument('--file', type=str, required=True, help='Path to the history .pkl.lz4 file.')
    parser.add_argument('--output', type=str, help='Output page path. Defaults to <history>.page.pdf.')
    parser.add_argument('--epoch', type=int, default=-1,
                        help='Rollout drawn on the page. A replay has one epoch, so this changes '
                             'nothing there; a training run has many and the default takes the last, '
                             'i.e. the converged policy.')
    parser.add_argument('--zoom_steps', type=int, default=200, help='Steps covered by the activity raster.')
    parser.add_argument('--zoom_start', type=int, default=0, help='First step of the activity raster.')
    parser.add_argument('--window', type=int, default=0,
                        help='Steps summarised by one point of every curve. 0 picks n_steps // 100.')
    parser.add_argument('--smooth', type=int, default=1,
                        help='Rolling-mean window, in points, applied to the curves after binning.')
    parser.add_argument('--n_drl', type=int, help='Number of learning stations. Defaults to the value in the history.')
    parser.add_argument('--dpi', type=int, default=200, help='Raster resolution of the saved page.')
    args = parser.parse_args()

    mpl.use('Agg')
    output = args.output or f"{args.file.removesuffix('.pkl.lz4')}.page.pdf"
    print(f"Saved: {build_page(args.file, output, args.epoch, args.zoom_steps, args.zoom_start, args.smooth, args.n_drl, args.window, args.dpi)}")
