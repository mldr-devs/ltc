"""Compare a trained agent against its distilled counterparts.

All inputs are ``ltc.run`` history files (``*.pkl.lz4``): the trained teacher
(``--agent_type ddqn``/``expected-sarsa``, passed as ``--trained``) and one or
both distillates produced by replaying it -- the random forest through the
``Forester`` agent (``--agent_type forester``, passed as ``--forester``) and the
symbolic expression through ``SRJaxAgent`` (``--agent_type sr-jax``, passed as
``--sr``). The script overlays their aggregate throughput and Jain's fairness
over time, reports the steady-state values side by side, and writes a small
summary. Give at least one distillate; two-way ``--distilled`` still works as an
alias for ``--sr``.

Built from ``plots_paper_traffic.py``: it reuses the same history-loading mocks,
the same steady-state metric definition, and the same window-aggregated time
series, but compares a handful of named runs instead of sweeping methods and
station counts.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import sys
import types
import csv
import shutil
import importlib.abc
import importlib.machinery
from argparse import ArgumentParser
from enum import IntEnum

import cloudpickle
import lz4.frame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Mocks and Patches ---------------------------------------------------------
# The histories pickle references classes from the training package. We stub the
# heavy ones so the file loads without importing jax / reinforced_lib / the agents.


class MockState:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockOutput:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)


class MockCarry:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)


if 'ltc.utils.scan_states' in sys.modules:
    del sys.modules['ltc.utils.scan_states']

scan_states = types.ModuleType('ltc.utils.scan_states')
scan_states.Output = MockOutput
scan_states.Carry = MockCarry
sys.modules['ltc.utils.scan_states'] = scan_states

ebaloha = types.ModuleType('ltc.agents.eb_aloha')
ebaloha.EBALOHA = MockState
ebaloha.EBALOHAState = MockState
sys.modules['ltc.agents.eb_aloha'] = ebaloha

tdma = types.ModuleType('ltc.agents.tdma')
tdma.TDMAState = MockState
tdma.TDMA = MockState
sys.modules['ltc.agents.tdma'] = tdma

slql = types.ModuleType('ltc.agents.StatelessQLearning')
slql.StatelessQLearningAgent = MockState
sys.modules['ltc.agents.StatelessQLearning'] = slql

try:
    import reinforced_lib.agents  # noqa: F401
except ImportError:
    rl_agents = types.ModuleType('reinforced_lib.agents')
    rl_agents.AgentState = MockState
    sys.modules['reinforced_lib.agents'] = rl_agents
    sys.modules['reinforced_lib'] = types.ModuleType('reinforced_lib')
    sys.modules['reinforced_lib'].agents = rl_agents


class _MockModule(types.ModuleType):
    """A module whose every attribute is MockState, for unpickling old classes."""
    def __getattr__(self, name):
        return MockState


class _MockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last-resort import hook: synthesize any missing agent/baseline module.

    Histories pickled under older commits reference classes from modules that have
    since moved or been renamed (e.g. ``ltc.agents.dcf`` now lives in
    ``ltc.baselines``). Appended to the *end* of ``sys.meta_path``, this only fires
    when the real module cannot be found, so present modules load normally.
    """
    PREFIXES = ('ltc.', 'reinforced_lib.')

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _MockModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.append(_MockFinder())


# --- Constants -----------------------------------------------------------------

class Actions(IntEnum):
    TX = 0
    CS = 1
    IDLE = 2


TRAINED_LABEL = 'Trained (KISS)'
FOREST_LABEL = 'Distilled (Forest)'
SR_LABEL = 'Distilled (SR)'

# Series render in this order; only those with a history provided are drawn.
SERIES_ORDER = [TRAINED_LABEL, FOREST_LABEL, SR_LABEL]

COLOR_MAP = {
    TRAINED_LABEL: 'red',
    FOREST_LABEL: 'C2',
    SR_LABEL: 'C0',
}

# --- Plotting Configuration ----------------------------------------------------

COLUMN_WIDTH = 4.3
COLUMN_HIGHT = 2 * COLUMN_WIDTH / (1 + 5 ** 0.5)

PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}

if not shutil.which('latex'):
    PLOT_PARAMS['text.usetex'] = False

plt.rcParams.update(PLOT_PARAMS)


# --- History loading and metrics ----------------------------------------------

def load_data(file_path):
    """Return the Output (all_outputs) object from an ltc.run history dump."""
    with lz4.frame.open(file_path, 'rb') as f:
        data = cloudpickle.load(f)
        if isinstance(data, tuple):
            for item in data:
                if hasattr(item, 'actions') or isinstance(item, MockOutput):
                    return item
                if isinstance(item, dict) and 'actions' in item:
                    return item
            return data[1]
        return data


def _flatten(history):
    """Return (actions, buffer_before, channel) flattened to (T, n_agents) / (T,)."""
    actions = np.array(history.actions)
    buffer = np.array(history.buffer_states)
    channel = np.array(history.channel_state)

    if actions.ndim == 3:
        n_agents = actions.shape[2]
        actions = actions.reshape(-1, n_agents)
        buffer = buffer.reshape(-1, n_agents)
        channel = channel.reshape(-1)
    elif actions.ndim == 1:
        actions = actions.reshape(-1, 1)
        buffer = buffer.reshape(-1, 1)

    buffer_before = np.zeros_like(buffer)
    buffer_before[1:] = buffer[:-1]
    return actions, buffer_before, channel


def per_agent_success(history):
    """Per-step, per-agent successful-transmission mask, shape (T, n_agents)."""
    actions, buffer_before, channel = _flatten(history)
    return (actions == Actions.TX.value) & (channel[:, None] == 1) & (buffer_before == 1)


def steady_state_metrics(history, last_percent=0.1):
    """Aggregate throughput and Jain's fairness over the final ``last_percent``."""
    success = per_agent_success(history)
    total_steps, n_agents = success.shape
    start = int(total_steps * (1 - last_percent))

    tail = success[start:]
    agg_throughput = tail.sum(axis=1).mean()

    agent_throughput = tail.mean(axis=0)
    denom = n_agents * (agent_throughput ** 2).sum()
    fairness = 0.0 if denom == 0 else (agent_throughput.sum() ** 2) / denom
    return float(agg_throughput), float(fairness)


def throughput_series(history, window_agg):
    """Window-averaged aggregate throughput and its step axis."""
    success = per_agent_success(history)
    agg = success.sum(axis=1)
    n_w = len(agg) // window_agg
    if n_w == 0:
        return np.array([]), np.array([])
    agg = agg[:n_w * window_agg].reshape(n_w, window_agg).mean(axis=1)
    xs = np.arange(n_w) * window_agg
    return xs, agg


def fairness_series(history, window_agg):
    """Window-averaged Jain's fairness index and its step axis."""
    success = per_agent_success(history)
    n_w = len(success) // window_agg
    if n_w == 0:
        return np.array([]), np.array([])
    reshaped = success[:n_w * window_agg].reshape(n_w, window_agg, -1)
    agent_thr = reshaped.mean(axis=1)
    sum_x = agent_thr.sum(axis=1)
    sum_sq = (agent_thr ** 2).sum(axis=1)
    n_agents = agent_thr.shape[1]
    denom = n_agents * sum_sq
    jains = np.divide(sum_x ** 2, denom, out=np.zeros_like(sum_x), where=denom != 0)
    xs = np.arange(n_w) * window_agg
    return xs, jains


# --- Plots ---------------------------------------------------------------------

def _thousands(x, _pos):
    return f'{int(x / 1000)}k' if x >= 1000 else str(int(x))


def plot_time_series(series, ylabel, filename, ylim):
    fig, ax = plt.subplots()
    for label, (xs, ys) in series.items():
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, label=label, color=COLOR_MAP.get(label, 'black'))
    ax.set_xlabel('Steps')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_summary_bars(summary, filename):
    labels = list(summary.keys())
    thr = [summary[l]['throughput'] for l in labels]
    fair = [summary[l]['fairness'] for l in labels]
    colors = [COLOR_MAP.get(l, 'black') for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, COLUMN_HIGHT))
    ax1.bar(labels, thr, color=colors)
    ax1.set_ylabel('Aggregate throughput')
    ax1.set_ylim(0, None)
    ax2.bar(labels, fair, color=colors)
    ax2.set_ylabel("Jain's fairness index")
    ax2.set_ylim(0, 1.1)
    for ax in (ax1, ax2):
        ax.tick_params(axis='x', labelrotation=20)
        ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


# --- Main ----------------------------------------------------------------------

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--trained', required=True, help='Trained-agent history (*.pkl.lz4).')
    parser.add_argument('--forester', help='Distilled random-forest (Forester) replay history (*.pkl.lz4).')
    parser.add_argument('--sr', help='Distilled symbolic-regression (SR-jax) replay history (*.pkl.lz4).')
    # Kept for backwards compatibility with the two-way ltc1_sr_debug wiring.
    parser.add_argument('--distilled', help='Alias for --sr.')
    parser.add_argument('--output_dir', default='out/compare', help='Directory for the comparison plots.')
    parser.add_argument('--window_agg', type=int, default=500, help='Steps averaged per time-series point.')
    parser.add_argument('--last_percent', type=float, default=0.1,
                        help='Fraction of the run used for the steady-state metrics.')
    args = parser.parse_args()

    sr_path = args.sr or args.distilled
    # Ordered {label: path}: trained teacher first, then whichever distillates were given.
    paths = {TRAINED_LABEL: args.trained}
    if args.forester:
        paths[FOREST_LABEL] = args.forester
    if sr_path:
        paths[SR_LABEL] = sr_path

    if len(paths) < 2:
        parser.error('Provide at least one distilled history (--forester and/or --sr).')

    os.makedirs(args.output_dir, exist_ok=True)

    histories = {}
    for label in SERIES_ORDER:
        if label in paths:
            print(f'Loading {label:<18}: {paths[label]}')
            histories[label] = load_data(paths[label])

    thr_series = {label: throughput_series(h, args.window_agg) for label, h in histories.items()}
    fair_series = {label: fairness_series(h, args.window_agg) for label, h in histories.items()}

    summary = {}
    for label, h in histories.items():
        thr, fair = steady_state_metrics(h, args.last_percent)
        summary[label] = {'throughput': thr, 'fairness': fair}

    plot_time_series(
        thr_series, 'Aggregate throughput',
        os.path.join(args.output_dir, 'throughput_vs_time.pdf'), ylim=(0, None),
    )
    plot_time_series(
        fair_series, "Jain's fairness index",
        os.path.join(args.output_dir, 'fairness_vs_time.pdf'), ylim=(0, 1.1),
    )
    plot_summary_bars(summary, os.path.join(args.output_dir, 'steady_state_summary.pdf'))

    # Numeric summary: printed and written to CSV next to the plots.
    print(f'\nSteady-state metrics (last {args.last_percent:.0%} of the run):')
    print(f'  {"":<20}{"throughput":>14}{"fairness":>12}')
    for label, m in summary.items():
        print(f'  {label:<20}{m["throughput"]:>14.4f}{m["fairness"]:>12.4f}')

    # Each distillate's throughput as a fraction of the trained teacher's.
    teacher = summary[TRAINED_LABEL]['throughput']
    if teacher:
        print()
        for label, m in summary.items():
            if label == TRAINED_LABEL:
                continue
            rel = 100 * (m['throughput'] - teacher) / teacher
            print(f'  {label} retains {100 * m["throughput"] / teacher:.1f}% of the trained '
                  f'throughput ({rel:+.1f}% relative).')

    csv_path = os.path.join(args.output_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent', 'throughput', 'fairness'])
        for label, m in summary.items():
            writer.writerow([label, m['throughput'], m['fairness']])

    print(f'\nWrote comparison plots and {csv_path} to {args.output_dir}/')


if __name__ == '__main__':
    main()
