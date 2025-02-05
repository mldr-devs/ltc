import matplotlib.pyplot as plt
import numpy as np

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


def plot_rewards(rewards, n, n_drl, seed, aggregation=100):
    plt.rcParams.update(PLOT_PARAMS)

    n_steps = rewards.shape[0]
    xs = np.linspace(0, n_steps - 1, n_steps // aggregation)

    rewards = rewards.reshape(-1, aggregation, n)
    rewards = rewards.mean(axis=1)

    for i in range(n_drl):
        plt.plot(xs, rewards[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, rewards[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.xlim(0, n_steps)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'rewards_{n}_{n_drl}_{seed}.pdf')
    plt.show()


def plot_cumulative_rewards(rewards, n, n_drl, seed, aggregation=100):
    plt.rcParams.update(PLOT_PARAMS)

    n_steps = rewards.shape[0]
    xs = np.linspace(0, n_steps - 1, n_steps // aggregation)

    cum_rewards = rewards.cumsum(axis=0)
    cum_rewards = cum_rewards.reshape(-1, aggregation, n)
    cum_rewards = cum_rewards.mean(axis=1)

    for i in range(n_drl):
        plt.plot(xs, cum_rewards[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, cum_rewards[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Step')
    plt.ylabel('Cumulative reward')
    plt.xlim(0, n_steps)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'cum_rewards_{n}_{n_drl}_{seed}.pdf')
    plt.show()


def plot_successful_transmissions(actions, channel_states, n, n_drl, seed, aggregation=100):
    plt.rcParams.update(PLOT_PARAMS)

    n_steps = actions.shape[0]
    xs = np.linspace(0, n_steps - 1, n_steps // aggregation)

    actions[np.where(channel_states != 1), :] = 0
    actions = actions.cumsum(axis=0)
    actions = actions.reshape(-1, aggregation, n)
    actions = actions.mean(axis=1)

    for i in range(n_drl):
        plt.plot(xs, actions[:, i], color='red')

    for i in range(n_drl, n):
        plt.plot(xs, actions[:, i], color='blue', linestyle='--')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Step')
    plt.ylabel('Successful transmissions')
    plt.xlim(0, n_steps)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'tx_{n}_{n_drl}_{seed}.pdf')
    plt.show()


def plot_dcf_collision_probabilities(channel_states, seed):
    analytical = [0, 0.103, 0.176, 0.23, 0.268, 0.301, 0.328, 0.353, 0.368, 0.382]

    n, vals = list(channel_states.keys()), []

    for i in n:
        tx_successful = (channel_states[i] == 1).sum()
        tx_failed = (channel_states[i] == -1).sum()
        vals.append(tx_failed / (tx_successful + tx_failed))

    plt.rcParams.update(PLOT_PARAMS)
    plt.plot(n, vals, marker='o', markersize=3, label='LTC DCF')
    plt.plot(n, analytical, linestyle='--', color='gray', label='Analytical')

    plt.xlabel('Number of DCF agents')
    plt.ylabel('Collision probability')
    plt.ylim(0, 0.5)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'collision_{seed}.pdf')
    plt.show()


def plot_cw_values(backoff, actions, n, n_drl, seed):
    cw = [0] + [2 ** i for i in range(4, 11)]
    counter = backoff[1:][actions[:-1] == 1]
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
    plt.savefig(f'cw_{n}_{n_drl}_{seed}.pdf')
    plt.show()
