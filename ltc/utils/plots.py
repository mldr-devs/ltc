from argparse import ArgumentParser

import cloudpickle
import lz4.frame
import matplotlib.pyplot as plt
import numpy as np

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


def plot_powers(power_states, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = power_states.shape[0]
    xs = np.arange(n_epochs) + 1

    consumed_power = (INITIAL_CAPACITY - power_states[:, -1]) / INITIAL_CAPACITY

    for i in range(n_drl):
        plt.plot(xs, consumed_power[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, consumed_power[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch')
    plt.ylabel('Consumed power')
    plt.xlim(1, n_epochs)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'power_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'power_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_rewards(rewards, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = rewards.shape[0]
    xs = np.arange(n_epochs) + 1

    rewards = rewards.mean(axis=1)

    for i in range(n_drl):
        plt.plot(xs, rewards[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, rewards[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.xlim(1, n_epochs)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'rewards_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'rewards_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_cumulative_rewards(rewards, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = rewards.shape[0]
    xs = np.arange(n_epochs) + 1

    cum_rewards = rewards.cumsum(axis=0)
    cum_rewards = cum_rewards.mean(axis=1)

    for i in range(n_drl):
        plt.plot(xs, cum_rewards[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, cum_rewards[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch')
    plt.ylabel('Cumulative reward')
    plt.xlim(1, n_epochs)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'cum_rewards_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'cum_rewards_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_successful_transmissions(actions, channel_states, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = actions.shape[0]
    xs = np.arange(n_epochs) + 1

    actions = actions.at[*np.where(channel_states != 1), :].set(0)
    actions = actions.sum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, actions[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, actions[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch')
    plt.ylabel('Successful transmissions')
    plt.xlim(1, n_epochs)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'tx_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'tx_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_channel_states(channel_states, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = channel_states.shape[0]
    xs = np.arange(n_epochs) + 1

    success = np.where(channel_states == 1, 1, 0)
    success = success.mean(axis=1)

    collision = np.where(channel_states == -1, 1, 0)
    collision = collision.mean(axis=1)

    idle = np.where(channel_states == 0, 1, 0)
    idle = idle.mean(axis=1)

    plt.plot(xs, success, color='green', label='Success')
    plt.plot(xs, collision, color='red', label='Collision')
    plt.plot(xs, idle, color='blue', label='Idle')

    plt.xlabel('Epoch')
    plt.ylabel('Channel state')
    plt.xlim(1, n_epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'channel_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'channel_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_channel_states_fill(channel_states, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = channel_states.shape[0]
    xs = np.arange(n_epochs) + 1

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

    plt.xlabel('Epoch')
    plt.ylabel('Channel state')
    plt.xlim(1, n_epochs)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'channel_fill_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'channel_fill_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_throughput(rewards, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = rewards.shape[0]
    xs = np.arange(n_epochs) + 1

    throughput = (rewards > NO_TX_REWARD)
    throughput = throughput.mean(axis=1)

    for i in range(n_drl):
        plt.plot(xs, throughput[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, throughput[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Epoch')
    plt.ylabel('Throughput')
    plt.xlim(1, n_epochs)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'throughput_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'throughput_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_throughput_fill(rewards, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = rewards.shape[0]
    xs = np.arange(n_epochs) + 1

    throughput = (rewards > NO_TX_REWARD)
    throughput = throughput.mean(axis=1).cumsum(axis=1)
    throughput = throughput / throughput[:, -1][:, None]

    for i in range(n_drl):
        plt.plot(xs, throughput[:, i], color='gray')

    for i in range(n_drl, n):
        plt.plot(xs, throughput[:, i], color='gray')

    plt.fill_between(xs, np.zeros_like(xs), throughput[:, n_drl - 1], color='red', alpha=0.3, label='DRL', linewidth=0)
    plt.fill_between(xs, throughput[:, n_drl - 1], throughput[:, -1], color='blue', alpha=0.3, label='DCF', linewidth=0)

    plt.xlabel('Epoch')
    plt.ylabel('Throughput')
    plt.xlim(1, n_epochs)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 6), [rf'{100 * i:.0f}\%' for i in np.linspace(0, 1, 6)])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'throughput_fill_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'throughput_fill_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_throughput_fill_nn(rewards, n, n_drl, seed):
    plt.rcParams.update(PLOT_PARAMS)

    n_epochs = rewards.shape[0]
    xs = np.arange(n_epochs) + 1

    throughput = (rewards > NO_TX_REWARD)
    throughput = throughput.mean(axis=1).cumsum(axis=1)

    for i in range(n_drl):
        plt.plot(xs, throughput[:, i], color='gray')

    for i in range(n_drl, n):
        plt.plot(xs, throughput[:, i], color='gray')

    plt.fill_between(xs, np.zeros_like(xs), throughput[:, n_drl - 1], color='red', alpha=0.3, label='DRL', linewidth=0)
    plt.fill_between(xs, throughput[:, n_drl - 1], throughput[:, -1], color='blue', alpha=0.3, label='DCF', linewidth=0)

    plt.xlabel('Epoch')
    plt.ylabel('Throughput')
    plt.xlim(1, n_epochs)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'throughput_fill_nn_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'throughput_fill_nn_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_all(filename):
    with lz4.frame.open(filename, 'rb') as f:
        _, history = cloudpickle.load(f)

    _, n, n_drl, seed_r, *_ = filename.split('_')
    seed, *_ = seed_r.split('.')
    n, n_drl, seed = int(n), int(n_drl), int(seed)

    plot_powers(history.power_states, n, n_drl, seed)
    plot_rewards(history.rewards, n, n_drl, seed)
    plot_cumulative_rewards(history.rewards, n, n_drl, seed)
    plot_successful_transmissions(history.actions, history.channel_state, n, n_drl, seed)
    plot_channel_states(history.channel_state, n, n_drl, seed)
    plot_channel_states_fill(history.channel_state, n, n_drl, seed)
    plot_throughput(history.rewards, n, n_drl, seed)
    plot_throughput_fill(history.rewards, n, n_drl, seed)
    plot_throughput_fill_nn(history.rewards, n, n_drl, seed)


def plot_dcf_collision_probabilities(actions, rewards, seed):
    analytical = [0, 0.103, 0.176, 0.23, 0.268, 0.301, 0.328, 0.353, 0.368, 0.382]

    n, vals = list(rewards.keys()), []

    for i in n:
        tx_successful = (rewards[i] > NO_TX_REWARD).sum(axis=0)
        tx_all = (actions[i] == Actions.TX.value).sum(axis=0)
        vals.append(1 - (tx_successful / tx_all).mean())

    plt.rcParams.update(PLOT_PARAMS)
    plt.plot(n, vals, marker='o', markersize=3, label='LTC DCF')
    plt.plot(n, analytical, linestyle='--', color='gray', label='Analytical')

    plt.xlabel('Number of DCF agents')
    plt.ylabel('Collision probability')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'collision_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'collision_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


def plot_cw_values(backoff, actions, n, n_drl, seed):
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
    plt.savefig(f'cw_{n}_{n_drl}_{seed}.pdf', bbox_inches='tight')
    plt.savefig(f'cw_{n}_{n_drl}_{seed}.png', bbox_inches='tight', dpi=300)
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('filename', type=str)
    args = args.parse_args()

    plot_all(args.filename)
