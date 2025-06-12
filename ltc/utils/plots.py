from argparse import ArgumentParser
from enum import Enum
from dataclasses import asdict

import cloudpickle
import jax
import lz4.frame
import matplotlib.pyplot as plt
import numpy as np

from ltc.utils.scan_states import Output
from ltc.sim.constants import NO_TX_REWARD, Actions, INITIAL_CAPACITY

COLUMN_WIDTH = 4.0
COLUMN_HIGHT = COLUMN_WIDTH * 0.618

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
    'lines.linewidth': 0.5,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


class PlotType(Enum):
    ALL = 'all'
    FIRST = 'first'


def plot_powers(power_states, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = power_states.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    consumed_power = (INITIAL_CAPACITY - power_states[:, -1]) / INITIAL_CAPACITY

    for i in range(n_drl):
        plt.plot(xs, consumed_power[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, consumed_power[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Consumed power')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(-0.05, 1.05)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_power_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_power_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_rewards(rewards, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = rewards.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    rewards = rewards.sum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, rewards[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, rewards[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Reward')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_rewards_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_rewards_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_cumulative_rewards(rewards, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = rewards.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    cum_rewards = rewards.cumsum(axis=0)
    cum_rewards = cum_rewards.sum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, cum_rewards[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, cum_rewards[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Cumulative reward')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_cum_rewards_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_cum_rewards_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_successful_transmissions(actions, channel_states, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = actions.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    actions = actions.at[*np.where(channel_states != 1), :].set(-1)
    actions = actions == Actions.TX.value
    actions = actions.sum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, actions[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, actions[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Successful transmissions')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_tx_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_tx_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_actions(actions, terminals, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = actions.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    tx = np.where((1 - terminals) * (actions == Actions.TX.value), 1, 0)
    tx = tx.sum(axis=1)
    tx_drl = tx[:, :n_drl] / (1 - terminals).sum(axis=1)[:, :n_drl]
    tx_dcf = tx[:, n_drl:] / (1 - terminals).sum(axis=1)[:, n_drl:]

    cs = np.where((1 - terminals) * (actions == Actions.CS.value), 1, 0)
    cs = cs.sum(axis=1)
    cs_drl = cs[:, :n_drl] / (1 - terminals).sum(axis=1)[:, :n_drl]
    cs_dcf = cs[:, n_drl:] / (1 - terminals).sum(axis=1)[:, n_drl:]

    idle = np.where((1 - terminals) * (actions == Actions.IDLE.value), 1, 0)
    idle = idle.sum(axis=1)
    idle_drl = idle[:, :n_drl] / (1 - terminals).sum(axis=1)[:, :n_drl]
    idle_dcf = idle[:, n_drl:] / (1 - terminals).sum(axis=1)[:, n_drl:]

    plt.plot(xs, tx_drl, color='red', linestyle='-')
    plt.plot(xs, cs_drl, color='green', linestyle='-')
    plt.plot(xs, idle_drl, color='blue', linestyle='-')

    plt.plot(xs, tx_dcf, color='red', linestyle='--', alpha=0.5)
    plt.plot(xs, cs_dcf, color='green', linestyle='--', alpha=0.5)
    plt.plot(xs, idle_dcf, color='blue', linestyle='--', alpha=0.5)

    plt.plot([], color='red', label='TX')
    plt.plot([], color='green', label='CS')
    plt.plot([], color='blue', label='Idle')
    plt.plot([], color='k', linestyle='-', label='DRL')
    plt.plot([], color='k', linestyle='--', label='DCF', alpha=0.5)

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Action')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_action_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_action_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_buffer_states(buffer_states, terminals, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = buffer_states.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    buffer_states = np.where(1 - terminals, buffer_states, 0).sum(axis=1)
    buffer_states = buffer_states / (1 - terminals).sum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, buffer_states[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, buffer_states[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Mean buffer fill')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(-0.05, 1.05)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_buffer_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_buffer_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_channel_states(channel_states, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = channel_states.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    success = np.where(channel_states == 1, 1, 0)
    success = success.mean(axis=1)

    collision = np.where(channel_states == -1, 1, 0)
    collision = collision.mean(axis=1)

    idle = np.where(channel_states == 0, 1, 0)
    idle = idle.mean(axis=1)

    plt.plot(xs, success, color='green', label='Success')
    plt.plot(xs, collision, color='red', label='Collision')
    plt.plot(xs, idle, color='blue', label='Idle')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Channel state')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_channel_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_channel_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_channel_states_fill(channel_states, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = channel_states.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    collision = np.where(channel_states == -1, 1, 0)
    collision = collision.mean(axis=1)

    idle = np.where(channel_states == 0, 1, 0)
    idle = idle.mean(axis=1) + collision

    success = np.where(channel_states == 1, 1, 0)
    success = success.mean(axis=1) + idle

    all = [np.zeros_like(success), collision, idle, success, np.ones_like(success)]

    plt.plot(xs, collision, color='k')
    plt.plot(xs, idle, color='k')
    plt.plot(xs, success, color='k')
    plt.fill_between(xs, all[0], all[1], color='red', alpha=0.3, label='Collision', linewidth=0)
    plt.fill_between(xs, all[1], all[2], color='blue', alpha=0.3, label='Idle', linewidth=0)
    plt.fill_between(xs, all[2], all[3], color='green', alpha=0.3, label='Success', linewidth=0)

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Channel state')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_channel_fill_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_channel_fill_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_throughput(actions, channel_states, terminals, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = actions.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    throughput = jax.lax.bitwise_and(actions == Actions.TX.value, channel_states[..., None] == 1)
    throughput = throughput.sum(axis=1) / (1 - terminals).sum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, throughput[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, throughput[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Throughput')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_throughput_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_throughput_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_throughput_fill(actions, channel_states, terminals, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = actions.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    throughput = jax.lax.bitwise_and(actions == Actions.TX.value, channel_states[..., None] == 1)
    throughput = (throughput.sum(axis=1) / (1 - terminals).sum(axis=1)).cumsum(axis=1)
    throughput = throughput / throughput[:, -1][:, None]

    for i in range(n_drl):
        plt.plot(xs, throughput[:, i], color='gray')

    for i in range(n_drl, n):
        plt.plot(xs, throughput[:, i], color='gray')

    plt.fill_between(xs, np.zeros_like(xs), throughput[:, n_drl - 1], color='red', alpha=0.3, label='DRL', linewidth=0)
    plt.fill_between(xs, throughput[:, n_drl - 1], throughput[:, -1], color='blue', alpha=0.3, label='DCF', linewidth=0)

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Throughput')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_throughput_fill_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_throughput_fill_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_throughput_fill_nn(actions, channel_states, terminals, n, n_drl, seed, name):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs, n_steps = actions.shape[:2]
    if name == PlotType.ALL:
        xs = np.arange(n_epochs) + 1
    else:
        xs = np.linspace(0, n_epochs * n_steps, n_epochs) + 1

    throughput = jax.lax.bitwise_and(actions == Actions.TX.value, channel_states[..., None] == 1)
    throughput = (throughput.sum(axis=1) / (1 - terminals).sum(axis=1)).cumsum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, throughput[:, i], color='gray')

    for i in range(n_drl, n):
        plt.plot(xs, throughput[:, i], color='gray')

    plt.fill_between(xs, np.zeros_like(xs), throughput[:, n_drl - 1], color='red', alpha=0.3, label='DRL', linewidth=0)
    plt.fill_between(xs, throughput[:, n_drl - 1], throughput[:, -1], color='blue', alpha=0.3, label='DCF', linewidth=0)

    plt.xlabel('Epoch' if name == PlotType.ALL else 'Step')
    plt.ylabel('Throughput')
    plt.xlim(1, n_epochs if name == PlotType.ALL else n_epochs * n_steps)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_throughput_fill_nn_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_throughput_fill_nn_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_all(filename):
    with lz4.frame.open(filename, 'rb') as f:
        _, history = cloudpickle.load(f)

    _, n, n_drl, seed_r, *_ = filename.split('_')
    seed, *_ = seed_r.split('.')
    n, n_drl, seed = int(n), int(n_drl), int(seed)

    plot_powers(history.power_states, n, n_drl, seed, PlotType.ALL)
    plot_rewards(history.rewards, n, n_drl, seed, PlotType.ALL)
    plot_cumulative_rewards(history.rewards, n, n_drl, seed, PlotType.ALL)
    plot_successful_transmissions(history.actions, history.channel_state, n, n_drl, seed, PlotType.ALL)
    plot_actions(history.actions, history.terminals, n, n_drl, seed, PlotType.ALL)
    plot_buffer_states(history.buffer_states, history.terminals, n, n_drl, seed, PlotType.ALL)
    plot_channel_states(history.channel_state, n, n_drl, seed, PlotType.ALL)
    plot_channel_states_fill(history.channel_state, n, n_drl, seed, PlotType.ALL)
    plot_throughput(history.actions, history.channel_state, history.terminals, n, n_drl, seed, PlotType.ALL)
    plot_throughput_fill(history.actions, history.channel_state, history.terminals, n, n_drl, seed, PlotType.ALL)
    plot_throughput_fill_nn(history.actions, history.channel_state, history.terminals, n, n_drl, seed, PlotType.ALL)


def plot_first(filename, n_epochs=10, aggregation=500):
    with lz4.frame.open(filename, 'rb') as f:
        _, history = cloudpickle.load(f)
        history = jax.tree.map(lambda x: x[:n_epochs], asdict(history))
        history = jax.tree.map(lambda x: x.reshape(-1, aggregation, *x.shape[2:]), history)
        history = Output(**history)

    _, n, n_drl, seed_r, *_ = filename.split('_')
    seed, *_ = seed_r.split('.')
    n, n_drl, seed = int(n), int(n_drl), int(seed)

    plot_powers(history.power_states, n, n_drl, seed, PlotType.FIRST)
    plot_rewards(history.rewards, n, n_drl, seed, PlotType.FIRST)
    plot_cumulative_rewards(history.rewards, n, n_drl, seed, PlotType.FIRST)
    plot_successful_transmissions(history.actions, history.channel_state, n, n_drl, seed, PlotType.FIRST)
    plot_actions(history.actions, history.terminals, n, n_drl, seed, PlotType.FIRST)
    plot_buffer_states(history.buffer_states, history.terminals, n, n_drl, seed, PlotType.FIRST)
    plot_channel_states(history.channel_state, n, n_drl, seed, PlotType.FIRST)
    plot_channel_states_fill(history.channel_state, n, n_drl, seed, PlotType.FIRST)
    plot_throughput(history.actions, history.channel_state, history.terminals, n, n_drl, seed, PlotType.FIRST)
    plot_throughput_fill(history.actions, history.channel_state, history.terminals, n, n_drl, seed, PlotType.FIRST)
    plot_throughput_fill_nn(history.actions, history.channel_state, history.terminals, n, n_drl, seed, PlotType.FIRST)


def plot_dcf_collision_probabilities(actions, rewards, seed, name):
    analytical = [0, 0.103, 0.176, 0.23, 0.268, 0.301, 0.328, 0.353, 0.368, 0.382]

    n, vals = list(rewards.keys()), []

    for i in n:
        tx_successful = (rewards[i] > NO_TX_REWARD).sum(axis=0)
        tx_all = (actions[i] == Actions.TX.value).sum(axis=0)
        vals.append(1 - (tx_successful / tx_all).mean())

    plt.rcParams.update(PLOT_PARAMS)
    plt.plot(n, vals, label='LTC DCF')
    plt.plot(n, analytical, linestyle='--', color='gray', label='Analytical')

    plt.xlabel('Number of DCF agents')
    plt.ylabel('Collision probability')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{name.value}_collision_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_collision_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_cw_values(backoff, actions, n, n_drl, seed, name):
    cw = [0] + [2 ** i for i in range(4, 11)]
    counter = backoff[1:][actions[:-1] == Actions.TX.value]
    bins, _ = np.histogram(counter, bins=cw)
    cw = [f'[{a},{b})' for a, b in zip(cw, cw[1:])]

    plt.rcParams.update(PLOT_PARAMS)
    plt.bar(cw, bins)

    plt.yscale('log')
    plt.xlabel('Backoff value')
    plt.ylabel('Count')
    plt.xticks(cw, fontsize=7)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{name.value}_cw_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'{name.value}_cw_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('filename', type=str)
    args = args.parse_args()

    plot_all(args.filename)
    plot_first(args.filename)
