"""Where a distilled forest looks: Gini importance over the observation window.

The forest is fitted on ``window_size * len(Features)`` columns named
``<feature>_<slot>``, so its importances reshape into a grid and read as a heatmap
-- one row per feature, one column per slot of the observation window.

Two choices worth stating. The colour scale is logarithmic and shared by both
panels: on the bursty run a single cell holds 91% of the importance while the
saturated one spreads it from 0 to 0.07, so a shared linear scale would render the
saturated panel blank, and a scale per panel would invite a comparison it does not
support. And ``viridis`` is monotone in luminance, so greyscale print keeps the
ordering of the cells -- readable, though not identical to the colour version the
way an achromatic ramp would be: the luminance range drops from 1.00 to 0.79, so
adjacent steps separate about a fifth less once the colour is gone.

Writes a PDF and a pgfplots ``.tex`` carrying the same data.
"""

from argparse import ArgumentParser
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ltc.symbolic.history2csv import FEATURE_NAMES

# Readable names for the paper, in the order ltc.sim.constants.Features defines.
FEATURE_LABELS = {
    'buffer': 'Buffer',
    'channel': 'Channel',
    'ret_c': 'Retransmissions',
    'no_tx': r'Steps since TX:$n_{\mathrm{idle}}$',
    'action_tx': 'Own action: TX',
    'action_cs': 'Own action: CS',
}
CMAP = 'viridis'
# Decades of colour scale below the largest importance. A log scale anchored on the
# smallest positive value is at the mercy of one noise cell -- the bursty forest has
# an importance of 5e-9, which would stretch the ramp over 8 decades and spend most
# of it on nothing. Anything below the floor, zeros included, draws as blank.
DYNAMIC_RANGE_DECADES = 5.0

PLOT_PARAMS = {
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


def importance_grid(forest_paths):
    """``([n_features, window_size] importances, window_size)``, newest slot last.

    Several paths are repeats of the same experiment under different seeds and are
    averaged. The per-seed maximum is not a stable statistic once importance is
    spread -- on the testbed forests the top cell differs between seeds while the
    vectors correlate at 0.85 to 0.96 -- so the mean is what there is to plot.
    """
    n_features = len(FEATURE_NAMES)
    importances = np.stack([joblib.load(p).feature_importances_ for p in forest_paths])
    window_size = importances.shape[1] // n_features
    # Columns run slot-major (see history2csv.build_column_names), so the reshape
    # gives [slot, feature] and the transpose puts features on the y axis.
    return importances.mean(axis=0).reshape(window_size, n_features).T, window_size


def shared_scale(grids):
    """One log scale for every panel, floored a fixed number of decades down."""
    positive = np.concatenate([g[g > 0].ravel() for g in grids])
    vmax = max(g.max() for g in grids)
    return LogNorm(vmin=max(positive.min(), vmax / 10 ** DYNAMIC_RANGE_DECADES), vmax=vmax)


def draw_pdf(grids, labels, window_size, norm, output):
    plt.rcParams.update(PLOT_PARAMS)
    # Width follows the window: 20 slots in the space 10 need collides the titles
    # and runs the tick labels together.
    panel_width = 2.4 + 0.07 * window_size
    fig, axes = plt.subplots(
        1, len(grids), figsize=(panel_width * len(grids), 2.3),
        constrained_layout=True, sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, grid, label, tag in zip(axes, grids, labels, 'abcdefg'):
        mesh = ax.pcolormesh(
            np.arange(window_size + 1) - 0.5, np.arange(len(FEATURE_NAMES) + 1) - 0.5,
            np.clip(grid, norm.vmin, None), cmap=CMAP, norm=norm,
        )
        # At most ~11 ticks: every slot is unreadable once the window is long.
        ax.set_xticks(range(0, window_size, max(1, -(-window_size // 11))))
        ax.set_xlabel('Window slot (newest last)')
        ax.set_title(f'({tag}) {label}')
        ax.set_aspect('auto')

    # Once, not per axis: they share the y axis, so a second flip undoes the first.
    axes[0].set_ylim(len(FEATURE_NAMES) - 0.5, -0.5)
    axes[0].set_yticks(range(len(FEATURE_NAMES)))
    axes[0].set_yticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES])
    axes[0].set_ylabel('Observation feature')

    bar = fig.colorbar(mesh, ax=axes, pad=0.02)
    bar.set_label('Gini importance (log scale)')
    bar.outline.set_linewidth(0.5)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def pgf_colormap(samples=16):
    """The same ramp as a pgfplots colormap definition."""
    colours = mpl.colormaps[CMAP](np.linspace(0, 1, samples))[:, :3]
    stops = '; '.join(f'rgb=({r:.3f},{g:.3f},{b:.3f})' for r, g, b in colours)
    return f'colormap={{bwramp}}{{{stops}}}'


def draw_tex(grids, labels, window_size, norm, output, pdf_name):
    """pgfplots source for the same figure, as a LaTeX float with two subplots."""
    lo, hi = np.log10(norm.vmin), np.log10(norm.vmax)
    ticks = [t for t in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)]

    panels = []
    for grid, label, tag in zip(grids, labels, 'abcdefg'):
        rows = []
        for y, feature in enumerate(FEATURE_NAMES):
            for x in range(window_size):
                # log10 of the clipped value: pgfplots maps point meta linearly, so
                # the transform has to happen here for the scale to match the PDF.
                rows.append(f'{x} {y} {np.log10(max(grid[y, x], norm.vmin)):.4f}')
        first = tag == 'a'
        last = tag == 'abcdefg'[len(grids) - 1]
        # One shared colorbar: in a groupplot it belongs to a single axis, or every
        # panel grows its own and they collide with the neighbour.
        bar = ('    colorbar right,\n'
               # y dir=reverse on the group would flip the bar as well.
               '    colorbar style={y dir=normal, ylabel={Gini importance}, '
               f'ytick={{{",".join(str(t) for t in ticks)}}}, '
               f'yticklabels={{{",".join(f"$10^{{{t}}}$" for t in ticks)}}}}},\n') if last else ''
        panels.append(
            '\\nextgroupplot[\n'
            + bar
            + f'    title={{({tag}) {label}}},\n'
            '    xlabel={Window slot (newest last)},\n'
            + ('    ylabel={Observation feature},\n' if first else '')
            + f'    xtick={{{",".join(str(t) for t in range(0, window_size, max(1, -(-window_size // 11))))}}},\n'
            f'    ytick={{0,...,{len(FEATURE_NAMES) - 1}}},\n'
            + ('    yticklabels={' + ','.join(f'{{{FEATURE_LABELS[f]}}}' for f in FEATURE_NAMES) + '},\n'
               if first else '    yticklabels={},\n')
            + ']\n'
            # mesh/cols is required: pgfplots cannot infer the matrix shape, and
            # without it 'matrix plot*' aborts the build.
            f'\\addplot[matrix plot*, mesh/cols={window_size}, point meta=explicit] '
            'table[meta=v] {\n'
            'x y v\n' + '\n'.join(rows) + '\n};\n'
        )

    tex = f"""% Generated by ltc.utils.forest_importance -- do not edit by hand.
% Requires: \\usepackage{{pgfplots}} \\usepgfplotslibrary{{groupplots}}
%           \\pgfplotsset{{compat=1.18}}
% A rendered copy of this figure is in {pdf_name}.
\\begin{{figure*}}
  \\centering
  \\begin{{tikzpicture}}
    \\pgfplotsset{{
      {pgf_colormap()},
      every axis/.append style={{font=\\small, axis line style={{line width=0.4pt}}}},
    }}
    \\begin{{groupplot}}[
      group style={{group size={len(grids)} by 1, horizontal sep=1.1cm}},
      width=6.2cm, height=4.6cm,
      enlargelimits=false, axis on top,
      y dir=reverse,
      point meta min={lo:.4f}, point meta max={hi:.4f},
    ]
{chr(10).join(panels)}    \\end{{groupplot}}
  \\end{{tikzpicture}}
  \\caption{{Gini importance of the distilled random forest over the observation
    window: one row per feature, one column per window slot, the newest slot last.
    Importances are normalised to sum to one within each panel. The colour scale is
    logarithmic and shared by the panels, because the importance is concentrated in
    one regime and diffuse in the other, and a linear scale would leave the diffuse
    panel blank. It spans {DYNAMIC_RANGE_DECADES:.0f} decades below the largest
    value; cells at or under that floor, zeros included, are blank.}}
  \\label{{fig:forest-importance}}
\\end{{figure*}}
"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(tex)
    return output


if __name__ == '__main__':
    parser = ArgumentParser(description='Heatmap of a distilled forest\'s feature importance.')
    parser.add_argument('--forest', type=str, nargs='+', required=True,
                        help='One subplot per argument. Comma-separate several forests to '
                             'average them, which is what the seeds of one experiment are.')
    parser.add_argument('--label', type=str, nargs='+', required=True,
                        help='Subplot title per forest, in the same order.')
    parser.add_argument('--pdf', type=str, required=True, help='Output PDF.')
    parser.add_argument('--tex', type=str, required=True, help='Output pgfplots .tex.')
    args = parser.parse_args()

    if len(args.forest) != len(args.label):
        parser.error('--forest and --label must have the same length.')

    mpl.use('Agg')
    groups = [arg.split(',') for arg in args.forest]
    grids, windows = zip(*(importance_grid(g) for g in groups))
    labels = [f'{l} ({len(g)} seeds)' if len(g) > 1 else l
              for l, g in zip(args.label, groups)]
    if len(set(windows)) != 1:
        parser.error(f'Forests disagree on the window size: {windows}.')

    norm = shared_scale(grids)
    print(f'Saved: {draw_pdf(grids, labels, windows[0], norm, args.pdf)}')
    print(f'Saved: {draw_tex(grids, labels, windows[0], norm, args.tex, Path(args.pdf).name)}')
