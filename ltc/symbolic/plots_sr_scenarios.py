"""Online (trained) vs distilled (SR) agents across the paper scenarios.

Reads <results>/s<seed>/compare_{sr,forest}_<cell>/summary.csv, with cells named sat_n<N>, low_n<N>, mid_n<N>,
err<p>_n<N> and hid<density>_n<N>, and writes results.csv, table.md and one figure per scenario.
"""
import glob
import os
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

AGENTS = {'Trained (KISS)': 'Online (KISS)', 'Distilled (SR)': 'Distilled (SR)', 'Distilled (Forest)': 'Distilled (Forest)'}
COLORS = {'Online (KISS)': 'C3', 'Distilled (SR)': 'C0', 'Distilled (Forest)': 'C2'}
CELL = re.compile(r'^(sat|low|mid|err|hid)([0-9.]*)_n(\d+)$')


def load(results: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(results, 's*', 'compare_*', 'summary.csv')):
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(path)))[1:])
        scenario, param, n = CELL.match(os.path.basename(os.path.dirname(path)).split('_', 2)[2]).groups()
        df = pd.read_csv(path)
        df['agent'] = df['agent'].map(AGENTS)
        rows.append(df.assign(scenario=scenario, param=float(param) if param else np.nan, n=int(n), seed=seed))
    if not rows:
        raise SystemExit(f'No summary.csv under {results}/s*/compare_sr_*/')
    return pd.concat(rows, ignore_index=True).drop_duplicates(['scenario', 'param', 'n', 'seed', 'agent'])


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 0.0 if len(x) < 2 else float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(['scenario', 'param', 'n', 'agent'], dropna=False)
    out = g.agg(thr=('throughput', 'mean'), thr_ci=('throughput', ci95),
                jain=('fairness', 'mean'), jain_ci=('fairness', ci95), seeds=('seed', 'nunique')).reset_index()
    wide = df.pivot_table(index=['scenario', 'param', 'n', 'seed'], columns='agent', values='throughput', dropna=False)
    ratio = wide.drop(columns='Online (KISS)').div(wide['Online (KISS)'], axis=0).stack().rename('ratio')
    ratio = ratio.groupby(['scenario', 'param', 'n', 'agent'], dropna=False).agg(['mean', ci95])
    ratio = ratio.rename(columns={'mean': 'ratio', 'ci95': 'ratio_ci'}).reset_index()
    return out.merge(ratio, on=['scenario', 'param', 'n', 'agent'], how='left')


def table(agg: pd.DataFrame) -> str:
    lines = ['| scenario | param | N | agent | throughput | Jain | distilled / online thr | seeds |', '|---|---|---|---|---|---|---|---|']
    for r in agg.sort_values(['scenario', 'param', 'n', 'agent']).itertuples():
        param = '' if np.isnan(r.param) else f'{r.param:g}'
        ratio = '' if np.isnan(r.ratio) else f'{r.ratio:.2f} ± {r.ratio_ci:.2f}'
        lines.append(f'| {r.scenario} | {param} | {r.n} | {r.agent} | {r.thr:.3f} ± {r.thr_ci:.3f} | '
                     f'{r.jain:.3f} ± {r.jain_ci:.3f} | {ratio} | {r.seeds} |')
    return '\n'.join(lines) + '\n'


def panel(ax, sub, x, y, label_suffix='', style='-o'):
    for agent, color in COLORS.items():
        a = sub[sub.agent == agent].sort_values(x)
        if len(a):
            ax.errorbar(a[x], a[y], yerr=a[f'{y}_ci'], fmt=style, color=color, ms=3, capsize=2, lw=1,
                        label=f'{agent}{label_suffix}')


def figure(agg, scenario, x, xlabel, filename, split_param=False):
    sub = agg[agg.scenario.isin(scenario)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    groups = sub.groupby(['scenario', 'param'], dropna=False) if split_param else [(None, sub)]
    styles = iter(['-o', '--s', ':^', '-.d'])
    for key, g in groups:
        suffix = '' if key is None else f' ({key[0]}{"" if np.isnan(key[1]) else f" {key[1]:g}"})'
        style = next(styles)
        panel(axes[0], g, x, 'thr', suffix, style)
        panel(axes[1], g, x, 'jain', suffix, style)
    axes[0].set_ylabel('Aggregate throughput')
    axes[1].set_ylabel("Jain's fairness index")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.4)
    axes[0].set_ylim(bottom=0)
    axes[1].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    fig.savefig(filename.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Directory with s<seed>/compare_{sr,forest}_<cell>/summary.csv.')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
    agg = aggregate(df)
    agg.to_csv(os.path.join(args.output_dir, 'aggregate.csv'), index=False)
    with open(os.path.join(args.output_dir, 'table.md'), 'w') as f:
        f.write(table(agg))

    figure(agg, ['sat'], 'n', 'Number of stations', os.path.join(args.output_dir, 'saturated.pdf'))
    figure(agg, ['low'], 'n', 'Number of stations', os.path.join(args.output_dir, 'low.pdf'))
    figure(agg, ['mid'], 'n', 'Number of stations', os.path.join(args.output_dir, 'mid.pdf'))
    figure(agg, ['err'], 'n', 'Number of stations', os.path.join(args.output_dir, 'interference.pdf'), True)
    hid = agg[agg.scenario == 'hid'].assign(density=lambda d: d.param)
    sat20 = agg[(agg.scenario == 'sat') & (agg.n == 20)].assign(scenario='hid', density=1.0)
    figure(pd.concat([hid, sat20]), ['hid'], 'density', 'Visibility density', os.path.join(args.output_dir, 'hidden.pdf'))
    print(table(agg))
    print(f'Wrote results.csv, aggregate.csv, table.md and figures to {args.output_dir}')


if __name__ == '__main__':
    main()
