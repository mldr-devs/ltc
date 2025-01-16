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

    for i in range(n - n_drl):
        plt.plot(xs, rewards[:, i], color='blue', linestyle='--')

    for i in range(n - n_drl, n):
        plt.plot(xs, rewards[:, i], color='red')

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

    for i in range(n - n_drl):
        plt.plot(xs, cum_rewards[:, i], color='blue', linestyle='--')

    for i in range(n - n_drl, n):
        plt.plot(xs, cum_rewards[:, i], color='red')

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
    actions = actions.reshape(-1, aggregation, n)
    actions = actions.mean(axis=1)

    for i in range(n - n_drl):
        plt.plot(xs, actions[:, i], color='blue', linestyle='--')

    for i in range(n - n_drl, n):
        plt.plot(xs, actions[:, i], color='red')

    plt.plot([], color='blue', linestyle='--', label='DCF')
    plt.plot([], color='red', label='DRL')

    plt.xlabel('Step')
    plt.ylabel('Successful transmissions')
    plt.xlim(0, n_steps)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'tx_{n}_{n_drl}_{seed}.pdf')
    plt.show()
